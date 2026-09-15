"""PPG-DaLiA: raw extraction, per-kind fully-anomalous copies, load-time anomaly mixing
and feature extraction, plus the two DataSource variants (signal / features) that read
the result back through ml.dataset_list's registry."""

import pickle as pkl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common.config import SEED

from .common import DataSource

DATASET_URL = 'https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip'

RAW_SUBDIR = 'PPG_FieldStudy'
CLEAN_SUBDIR = 'clean-signals'
ANOMALOUS_SUBDIR = 'anomalous-signals'      # per-type fully-anomalous BVP: <kind>/S*/
ACTIVITY_FILE = 'activity.npy'

BVP_RATE = 64
ACC_RATE = 32
ACTIVITY_RATE = 4
WINDOW_SECONDS = 8
BVP_WINDOW = BVP_RATE * WINDOW_SECONDS    # 512 samples
ACC_WINDOW = ACC_RATE * WINDOW_SECONDS    # 256 samples
ANOMALY_PROB = 0.5
MIN_ANOMALY_WINDOWS = 8
MAX_ANOMALY_WINDOWS = 30

ANOMALY_KINDS = ('blowup', 'noise', 'tachy', 'brady', 'afib')
N_FEATURES = 20

# Pulse band for the three shape features, 0.5-4.0 Hz = 30-240 bpm. Deliberately wider
# than the HR-band ratio's 0.7-3.5 Hz so a slowed rhythm still falls inside it.
PULSE_BAND_LOW = 0.5
PULSE_BAND_HIGH = 4.0
FEATURE_EPS = 1e-9

# PPG-DaLiA's protocol activities, as stored in SX.pkl['activity'] at 4 Hz. ID 0 marks the
# transient periods between activities (mostly walking to the next location).
ACTIVITIES = {
    0: 'transient', 1: 'sitting', 2: 'stairs', 3: 'table-soccer', 4: 'cycling',
    5: 'driving', 6: 'lunch', 7: 'walking', 8: 'working',
}

# The activities a subject stays essentially still through. Anything else is dominated by
# motion artefacts, which swamp the waveform morphology an anomaly detector reads.
LOW_ACTIVITY = (1, 5, 6, 8)

CLEAN = 'clean'
MIXED = 'mixed'
VARIANTS = (CLEAN, MIXED, *ANOMALY_KINDS)

# Modality tags ml.dataset_list uses to build its registry keys.
SIGNAL = 'signal'
FEATURES = 'features'

# The non-overlapping grid every registry-built signal source uses by default — the one
# that aligns with labels() and with what scoring/export/plotting consume. Features have
# no independent windowing concept, so they always use this grid.
EVAL_WINDOW = BVP_WINDOW
EVAL_SHIFT = BVP_WINDOW
# The waveform autoencoders' denser, overlapping training grid (see AutoencoderTrainer in
# ml.models.signal) — more training data per subject than the eval grid gives.
TRAIN_SHIFT = BVP_RATE * 3


class DatasetUnavailibleError(FileNotFoundError):
    def __init__(self, data_dir: str | Path):
        self.message = f"Dataset not found at {data_dir}. Run scripts/get_dataset.py first."
        super().__init__(self.message)


def get_sorted_paths(dataset_dir: Path) -> list[Path]:
    dir_list = [d for d in dataset_dir.glob('S*') if d.is_dir() and d.name[1:].isdigit()]
    return sorted(dir_list, key=lambda d: int(d.name[1:]))


# ---------------------------------------------------------------------------
# Stage 1 — Extract raw signals
# ---------------------------------------------------------------------------

def upsample_activity(activity: np.ndarray, length: int) -> np.ndarray:
    """The 4 Hz activity track resampled onto the BVP sample grid, zero-padded on any short tail."""
    upsampled = np.repeat(activity.reshape(-1), BVP_RATE // ACTIVITY_RATE)[:length]
    pad = length - len(upsampled)
    if pad > 0:
        upsampled = np.concatenate([upsampled, np.zeros(pad, dtype=upsampled.dtype)])
    return upsampled.astype(np.uint8)


def extract_subject_signals(raw_dir: Path, subjects_dir: Path) -> list[int]:
    """Extract raw BVP (64 Hz), ACC magnitude (32 Hz) and the upsampled activity track per subject."""
    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_raw_dirs = get_sorted_paths(raw_dir)

    processed = []

    for subject_raw_dir in subject_raw_dirs:
        subject_dir_name = subject_raw_dir.name
        path = subject_raw_dir / f'{subject_dir_name}.pkl'
        raw = pkl.loads(path.read_bytes(), encoding='latin1')

        wrist = raw['signal']['wrist']
        bvp = wrist['BVP'].flatten().astype(np.float32)

        acc_g = wrist['ACC'] / 64.0
        acc = np.sqrt(np.sum(acc_g ** 2, axis=1)).astype(np.float32)

        activity = upsample_activity(np.asarray(raw['activity']), len(bvp))

        save_dir = subjects_dir / subject_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy', bvp)
        np.save(save_dir / 'acc.npy', acc)
        np.save(save_dir / ACTIVITY_FILE, activity)

        processed.append(subject_dir_name)

        low = float(np.isin(activity, LOW_ACTIVITY).mean())
        print(f"  {subject_dir_name}: BVP {len(bvp)} samples @ {BVP_RATE} Hz, "
              f"ACC {len(acc)} samples @ {ACC_RATE} Hz, {low:.1%} low-activity")

    return processed


# ---------------------------------------------------------------------------
# Stage 2 — Synthetic anomalies on raw BVP
# ---------------------------------------------------------------------------

def wavy_noise(n: int) -> np.ndarray:
    n_steps = n // BVP_RATE * 3
    noise = np.random.uniform(-1.0, 1.0, size=n_steps)
    wavy = np.fft.irfft(np.fft.rfft(noise), n)
    return wavy / (wavy.std() * 2)

def stretch_by(factor, x, y):
    m = int(round(len(x) * factor))
    return np.interp(np.linspace(0, len(x) - 1, m), x, y)

def apply_anomaly(segment: np.ndarray, kind: str) -> np.ndarray:
    """Return a perturbed copy of a BVP ``segment`` for ``ANOMALY_KINDS[kind]``."""
    seg = segment.copy()
    n = len(seg)
    src = np.linspace(0, n - 1, n)

    if kind == 'blowup':   # amplitude blow-up around the local mean
        mean = float(seg.mean())
        seg = mean + (seg - mean) * 1.7

    elif kind == 'noise':  # wavy band-limited interference burst
        seg += wavy_noise(n) * (seg.max() - seg.min()) / 15

    elif kind == 'tachy':  # increased tempo by shrinking and tiling
        resampled = stretch_by(0.7, src, seg)
        seg = np.tile(resampled, int(np.ceil(n / len(resampled))))[:n]

    elif kind == 'brady':  # decreased tempo by stretching
        resampled = stretch_by(1.7, src, seg)
        seg = resampled[:n]

    elif kind == 'afib':   # irregularly-irregular rhythm via a jittered monotonic warp
        win_count = n // BVP_RATE
        speed = np.interp(src, np.linspace(0, n - 1, win_count),
                          np.random.beta(0.8, 0.8, size=win_count) * 1.4 + 0.3)
        warp = np.cumsum(speed)
        warp *= (n - 1) / warp[-1]       # normalize to [0, n-1], endpoints fixed
        seg = np.interp(warp, src, seg)

    else:
        raise Exception(f"unknown anomaly kind {kind}")

    return seg.astype(np.float32)


def subject_rng(sid: str) -> np.random.Generator:
    """The RNG a subject's load-time anomaly mix is drawn from, keyed by subject id."""
    return np.random.default_rng([SEED, int(sid[1:])])


def mix_signal(bvp: np.ndarray, rng: np.random.Generator,
               anomaly_prob: float = ANOMALY_PROB) -> tuple[np.ndarray, np.ndarray]:
    """Inject a window-aligned mix of random anomaly kinds, returning (anomalous_bvp, win_labels)."""
    result = bvp.copy()
    n_windows = len(bvp) // BVP_WINDOW
    win_labels = np.zeros(max(n_windows, 0), dtype=np.float32)
    if n_windows == 0:
        return result.astype(np.float32), win_labels

    target = int(n_windows * anomaly_prob)

    while int(win_labels.sum()) < target:
        length = int(rng.integers(MIN_ANOMALY_WINDOWS, MAX_ANOMALY_WINDOWS + 1))
        start = int(rng.integers(0, n_windows - length + 1))

        wins = slice(start, start + length)
        if win_labels[wins].any():
            continue

        seg = slice(start * BVP_WINDOW, (start + length) * BVP_WINDOW)
        kind = ANOMALY_KINDS[int(rng.integers(len(ANOMALY_KINDS)))]
        result[seg] = apply_anomaly(result[seg], kind)

        win_labels[wins] = 1.0

    return result.astype(np.float32), win_labels


def inject_single_kind(bvp: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Apply one anomaly kind to every window of a raw BVP signal by tiling window-aligned spans across it"""
    result = bvp.copy()
    n_windows = len(bvp) // BVP_WINDOW
    if n_windows == 0:
        return result.astype(np.float32)

    w = 0
    while w < n_windows:
        length = min(int(rng.integers(MIN_ANOMALY_WINDOWS, MAX_ANOMALY_WINDOWS + 1)), n_windows - w)
        seg = slice(w * BVP_WINDOW, (w + length) * BVP_WINDOW)
        result[seg] = apply_anomaly(result[seg], kind)
        w += length

    return result.astype(np.float32)


def create_anomalous_signals(subjects_dir: Path, anomalous_dir: Path):
    """Per-type fully-anomalous BVP for isolated testing, written to ``<anomalous_dir>/<kind>/S*/bvp.npy``."""
    rng = np.random.default_rng(SEED)

    for kind in ANOMALY_KINDS:
        kind_dir = anomalous_dir / kind
        subject_dirs = get_sorted_paths(subjects_dir)
        for subject_dir in subject_dirs:
            sid = subject_dir.name
            bvp = np.load(subject_dir / 'bvp.npy')
            anomalous_bvp = inject_single_kind(bvp, kind, rng)
            save_dir = kind_dir / sid
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / 'bvp.npy', anomalous_bvp)
        print(f"  {kind}: {len(subject_dirs)} subjects")


def prepare_ppg_dalia(datasets_dir: Path) -> None:
    raw_dir = datasets_dir / RAW_SUBDIR
    subjects_dir = datasets_dir / CLEAN_SUBDIR
    anomalous_dir = datasets_dir / ANOMALOUS_SUBDIR

    if raw_dir.is_dir():
        print(f"Raw dataset already present at {raw_dir}")
    else:
        datasets_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=datasets_dir) as tmp:
            tmp_dir = Path(tmp)
            outer_zip = tmp_dir / 'ppg-dalia.zip'
            print(f"Downloading {DATASET_URL} ...")
            urllib.request.urlretrieve(DATASET_URL, outer_zip)

            with zipfile.ZipFile(outer_zip) as zf:
                zf.extractall(tmp_dir)

            inner_zip = tmp_dir / 'data.zip'
            print(f"Extracting dataset into {datasets_dir}/ ...")
            with zipfile.ZipFile(inner_zip) as zf:
                zf.extractall(datasets_dir)
        print(f"Raw dataset ready at {raw_dir}")

    if subjects_dir.is_dir():
        print(f"{CLEAN_SUBDIR} already present at {subjects_dir}")
    else:
        print(f"\nStage 1: Extracting raw signals into {subjects_dir}/ ...")
        written = extract_subject_signals(raw_dir, subjects_dir)
        print(f"Processed {len(written)} subjects")

    if anomalous_dir.is_dir() and any(anomalous_dir.glob('*/S*')):
        print(f"{ANOMALOUS_SUBDIR} already present at {anomalous_dir}")
    else:
        print(f"\nStage 2: Creating per-type anomalous signals in {anomalous_dir}/ ...")
        create_anomalous_signals(subjects_dir, anomalous_dir)


# ---------------------------------------------------------------------------
# Load-time feature extraction
# ---------------------------------------------------------------------------

def extract_features(bvp_window: np.ndarray, acc_window: np.ndarray) -> np.ndarray:
    """20-feature vector from an 8-second BVP window (512 samples) and ACC window (256 samples)"""
    feats: list[float] = []

    for ch in (bvp_window, acc_window):
        feats += [
            float(ch.mean()),
            float(ch.std()),
            float(ch.min()),
            float(ch.max()),
            float(ch.max() - ch.min()),
            float(np.sqrt(np.mean(ch ** 2))),
            float(np.mean(np.abs(np.diff(ch)))),
        ]

    bvp   = bvp_window - bvp_window.mean()
    signs = np.sign(bvp)
    feats.append(float(np.sum(np.abs(np.diff(signs)) > 0)) / (len(bvp) - 1))

    hann     = np.hanning(len(bvp))
    windowed = bvp * hann
    rfft     = np.fft.rfft(windowed)
    power    = rfft.real ** 2 + rfft.imag ** 2
    freqs    = np.fft.rfftfreq(len(bvp_window), d=1.0 / BVP_RATE)
    feats.append(float(freqs[np.argmax(power)]))
    band = (freqs >= 0.7) & (freqs <= 3.5)
    feats.append(float(power[band].sum() / (power.sum() + 1e-8)))

    total = power.sum() + FEATURE_EPS
    pulse = (freqs >= PULSE_BAND_LOW) & (freqs <= PULSE_BAND_HIGH)
    pulse_power, pulse_freqs = power[pulse], freqs[pulse]
    pulse_total = pulse_power.sum() + FEATURE_EPS

    centroid = float((pulse_power * pulse_freqs).sum() / pulse_total)
    variance = float((pulse_power * (pulse_freqs - centroid) ** 2).sum() / pulse_total)
    feats.append(centroid)
    feats.append(float(np.sqrt(max(variance, 0.0))))
    feats.append(float(np.log(power[freqs > PULSE_BAND_HIGH].sum() / total + FEATURE_EPS)))

    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# DataSource variants
# ---------------------------------------------------------------------------

class _DaliaBase:
    """Shared plumbing behind both Dalia DataSource variants: subject listing, the
    variant-aware raw BVP stream (clean / mixed / one fully-anomalous kind), the raw ACC
    stream (always the clean one — anomalies are injected into BVP only), window-grid
    arithmetic, the activity filter and per-subject z-score stats."""

    def __init__(self, data_root: Path, activities: tuple[int, ...] | None = None):
        self.data_root = data_root
        self.activities = activities
        self.clean_dir = data_root / CLEAN_SUBDIR
        self._mixed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._stats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def signal_dir(self, variant: str) -> Path:
        if variant == CLEAN:
            return self.clean_dir
        if variant in ANOMALY_KINDS:
            return self.data_root / ANOMALOUS_SUBDIR / variant
        raise ValueError(f"unknown signal variant {variant!r}, expected one of {VARIANTS}")

    def subject_ids(self) -> list[str]:
        dirs = get_sorted_paths(self.clean_dir)
        if not dirs:
            raise DatasetUnavailibleError(self.clean_dir)
        return [d.name for d in dirs]

    def _length(self, path: Path) -> int:
        return int(np.load(path, mmap_mode='r').shape[0])

    def n_windows(self, sid: str, window: int, shift: int) -> int:
        n_bvp = self._length(self.clean_dir / sid / 'bvp.npy')
        count = (n_bvp - window) // shift + 1 if n_bvp >= window else 0
        if window == BVP_WINDOW and shift == BVP_WINDOW:
            # The feature grid is also bounded by ACC, which runs at half the rate and
            # can end a window short (see extract_features' pairing of the two).
            n_acc = self._length(self.clean_dir / sid / 'acc.npy')
            count = min(count, (n_acc - ACC_WINDOW) // ACC_WINDOW + 1)
        return max(count, 0)

    def window_mask(self, sid: str, window: int, shift: int) -> np.ndarray:
        count = self.n_windows(sid, window, shift)
        if self.activities is None:
            return np.ones(count, dtype=bool)

        activity = np.load(self.clean_dir / sid / ACTIVITY_FILE)
        allowed = np.zeros(int(activity.max()) + 1, dtype=bool)
        for act in self.activities:
            if act < len(allowed):
                allowed[act] = True

        # Running count of disallowed samples, so a window is kept iff it contains none.
        excluded = np.concatenate([[0], np.cumsum(~allowed[activity])])
        starts = np.arange(count) * shift
        return (excluded[starts + window] - excluded[starts]) == 0

    def _mix(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        """The subject's mixed signal and its per-window labels, built once per source."""
        if sid not in self._mixed:
            self._mixed[sid] = mix_signal(self.raw_signal(sid, CLEAN), subject_rng(sid))
        return self._mixed[sid]

    def raw_signal(self, sid: str, variant: str) -> np.ndarray:
        if variant == MIXED:
            return self._mix(sid)[0]
        path = self.signal_dir(variant) / sid / 'bvp.npy'
        if not path.exists():
            raise DatasetUnavailibleError(self.signal_dir(variant))
        return np.load(path)

    def raw_acc(self, sid: str) -> np.ndarray:
        return np.load(self.clean_dir / sid / 'acc.npy')

    def _raw_windows(self, sid: str, variant: str, window: int, shift: int) -> np.ndarray:
        """``(n, window)`` un-normalized BVP windows on the kept grid."""
        signal = self.raw_signal(sid, variant)
        count = self.n_windows(sid, window, shift)
        if count == 0:
            return np.empty((0, window), dtype=np.float32)
        windows = sliding_window_view(signal, window)[::shift][:count]
        return windows[self.window_mask(sid, window, shift)].astype(np.float32)

    def raw_features_at(self, sid: str, variant: str, window: int, shift: int) -> np.ndarray:
        """``(n, N_FEATURES)`` un-normalized feature vectors on the given grid."""
        bvp = self._raw_windows(sid, variant, window, shift)
        if not len(bvp):
            return np.empty((0, N_FEATURES), dtype=np.float32)

        acc = self.raw_acc(sid)
        count = self.n_windows(sid, window, shift)
        acc_windows = np.stack([acc[i * ACC_WINDOW:(i + 1) * ACC_WINDOW]
                                for i in range(count)])
        acc_windows = acc_windows[self.window_mask(sid, window, shift)]

        return np.stack([extract_features(b, a) for b, a in zip(bvp, acc_windows)])

    def labels_on_grid(self, sid: str, window: int, shift: int) -> np.ndarray:
        """Per-window binary anomaly truth for the mixed variant, on the given grid."""
        labels = self._mix(sid)[1]
        count = min(len(labels), self.n_windows(sid, window, shift))
        return labels[:count][self.window_mask(sid, window, shift)[:count]]

    def clean_reference(self, sid: str, family: str) -> np.ndarray:
        if family == SIGNAL:
            return self.raw_signal(sid, CLEAN).reshape(-1, 1)
        if family == FEATURES:
            return self.raw_features_at(sid, CLEAN, EVAL_WINDOW, EVAL_SHIFT)
        raise ValueError(f"unknown normalization family {family!r}")

    def norm_stats(self, sid: str, family: str) -> tuple[np.ndarray, np.ndarray]:
        key = (sid, family)
        if key not in self._stats:
            reference = self.clean_reference(sid, family)
            self._stats[key] = (reference.mean(axis=0).astype(np.float32),
                                reference.std(axis=0).astype(np.float32) + 1e-6)
        return self._stats[key]

    def normalize(self, sid: str, family: str, values: np.ndarray) -> np.ndarray:
        mean, std = self.norm_stats(sid, family)
        return ((values - mean) / std).astype(np.float32)

    def sample(self, arrays: list[np.ndarray], per_subject: int) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        parts = [a[rng.choice(len(a), min(per_subject, len(a)), replace=False)]
                 for a in arrays if len(a)]
        if not parts:
            raise DatasetUnavailibleError(self.data_root)
        return np.concatenate(parts)


class DaliaSignalSource(DataSource):
    """One modality/variant of PPG-DaLiA's BVP signal: z-scored sliding windows on
    ``(window, shift)``, defaulting to the non-overlapping eval grid. ``labels`` and
    ``raw_signal``/``raw_acc`` (the latter two outside the generic interface, for the
    firmware/app export script) are only meaningful for the ``mixed`` variant / any
    variant respectively."""

    def __init__(self, data_root: Path, key: str, variant: str,
                 activities: tuple[int, ...] | None = None,
                 window: int = EVAL_WINDOW, shift: int = EVAL_SHIFT):
        self.key = key
        self.variant = variant
        self.window = window
        self.shift = shift
        self._base = _DaliaBase(data_root, activities)

    def subject_ids(self) -> list[str]:
        return self._base.subject_ids()

    def datapoints(self, sid: str) -> np.ndarray:
        raw = self._base._raw_windows(sid, self.variant, self.window, self.shift)
        raw = raw.reshape(-1, self.window, 1)
        return self._base.normalize(sid, SIGNAL, raw)

    def labels(self, sid: str) -> np.ndarray | None:
        if self.variant != MIXED:
            return None
        return self._base.labels_on_grid(sid, self.window, self.shift)

    def calibration_data(self, per_subject: int = 10) -> np.ndarray:
        return self._base.sample([self.datapoints(sid) for sid in self.subject_ids()],
                                 per_subject)

    def raw_signal(self, sid: str) -> np.ndarray:
        """The subject's whole raw BVP stream for this source's variant, unwindowed and
        unfiltered — what a sensor would emit. Only the export script wants this; a model
        is fed by ``datapoints``."""
        return self._base.raw_signal(sid, self.variant)

    def raw_acc(self, sid: str) -> np.ndarray:
        """The subject's whole raw ACC magnitude stream (always the clean one)."""
        return self._base.raw_acc(sid)

    def with_grid(self, window: int | None = None, shift: int | None = None) -> 'DaliaSignalSource':
        """The same data on a different ``(window, shift)`` grid."""
        return DaliaSignalSource(self._base.data_root, key=self.key, variant=self.variant,
                                 activities=self._base.activities,
                                 window=self.window if window is None else window,
                                 shift=self.shift if shift is None else shift)


class DaliaFeatureSource(DataSource):
    """One modality/variant of PPG-DaLiA's hand-crafted feature vectors, always on the
    non-overlapping window grid (features have no independent windowing concept)."""

    def __init__(self, data_root: Path, key: str, variant: str,
                 activities: tuple[int, ...] | None = None):
        self.key = key
        self.variant = variant
        self._base = _DaliaBase(data_root, activities)

    def subject_ids(self) -> list[str]:
        return self._base.subject_ids()

    def datapoints(self, sid: str) -> np.ndarray:
        raw = self._base.raw_features_at(sid, self.variant, EVAL_WINDOW, EVAL_SHIFT)
        return self._base.normalize(sid, FEATURES, raw)

    def labels(self, sid: str) -> np.ndarray | None:
        if self.variant != MIXED:
            return None
        return self._base.labels_on_grid(sid, EVAL_WINDOW, EVAL_SHIFT)

    def calibration_data(self, per_subject: int = 10) -> np.ndarray:
        return self._base.sample([self.datapoints(sid) for sid in self.subject_ids()],
                                 per_subject)

    def raw_features(self, sid: str) -> np.ndarray:
        """``(n, N_FEATURES)`` un-normalized feature vectors — the vectors as the
        firmware computes and reports them. Only the export script wants this; a model
        is fed by ``datapoints``."""
        return self._base.raw_features_at(sid, self.variant, EVAL_WINDOW, EVAL_SHIFT)
