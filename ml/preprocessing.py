"""Dataset processing for the PPG-DaLiA anomaly-detection pipeline: raw extraction,
per-kind fully-anomalous copies, anomaly injection, and the feature extractor.
Anomaly mix, features and normalization are computed at load time by ``ml.sources``."""

import pickle as pkl
from pathlib import Path
import numpy as np

from common.config import SEED

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
N_FEATURES = 17

# PPG-DaLiA's protocol activities, as stored in SX.pkl['activity'] at 4 Hz. ID 0 marks the
# transient periods between activities (mostly walking to the next location).
ACTIVITIES = {
    0: 'transient', 1: 'sitting', 2: 'stairs', 3: 'table-soccer', 4: 'cycling',
    5: 'driving', 6: 'lunch', 7: 'walking', 8: 'working',
}

# The activities a subject stays essentially still through. Anything else is dominated by
# motion artefacts, which swamp the waveform morphology an anomaly detector reads.
LOW_ACTIVITY = (1, 5, 6, 8)

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


# ---------------------------------------------------------------------------
# Load-time feature extraction
# ---------------------------------------------------------------------------

def extract_features(bvp_window: np.ndarray, acc_window: np.ndarray) -> np.ndarray:
    """17-feature vector from an 8-second BVP window (512 samples) and ACC window (256 samples)"""
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

    # Zero-crossing rate of mean-centred BVP (matches firmware sign-change loop)
    bvp   = bvp_window - bvp_window.mean()
    signs = np.sign(bvp)
    feats.append(float(np.sum(np.abs(np.diff(signs)) > 0)) / (len(bvp) - 1))

    # Spectral features: Hann window + power (magnitude²) ratios
    hann     = np.hanning(len(bvp))
    windowed = bvp * hann
    rfft     = np.fft.rfft(windowed)
    power    = rfft.real ** 2 + rfft.imag ** 2
    freqs    = np.fft.rfftfreq(len(bvp_window), d=1.0 / BVP_RATE)
    feats.append(float(freqs[np.argmax(power)]))
    band = (freqs >= 0.7) & (freqs <= 3.5)
    feats.append(float(power[band].sum() / (power.sum() + 1e-8)))

    return np.asarray(feats, dtype=np.float32)
