"""Dataset processing for the PPG-DaLiA anomaly-detection pipeline.

Owns everything between the raw download and the numpy arrays on disk: the
per-stage build functions (raw extraction, synthetic-anomaly injection, feature
extraction), the normalization params they save, and the shared constants.
``scripts/system/get_dataset.py`` is a thin CLI that downloads the archive and
sequences these stages. Deliberately free of TensorFlow — ``ml.loading`` builds
the tf.data pipelines on top of these arrays.
"""

import pickle as pkl
import random
from pathlib import Path
import numpy as np

from common.config import SEED

RAW_SUBDIR = 'PPG_FieldStudy'
CLEAN_SUBDIR = 'clean-signals'
ANOMALOUS_SUBDIR = 'anomalous-signals'      # per-type fully-anomalous BVP: <kind>/S*/
MIXED_SUBDIR = 'mixed-signals'              # realistic ~50% mix: S*/ (bvp + binary labels)
MIXED_FEATURE_SUBDIR = 'mixed-features'     # features windowed from mixed-signals
CLEAN_FEATURE_SUBDIR = 'clean-features'     # features windowed from clean-signals (all-normal)
NORM_PARAMS_FILE = 'norm-params.npy'
FEATURE_STATS_FILE = 'feature_stats.npy'

BVP_RATE = 64
ACC_RATE = 32
WINDOW_SECONDS = 8
BVP_WINDOW = BVP_RATE * WINDOW_SECONDS    # 512 samples
ACC_WINDOW = ACC_RATE * WINDOW_SECONDS    # 256 samples
ANOMALY_PROB = 0.5
MIN_ANOMALY_WINDOWS = 8
MAX_ANOMALY_WINDOWS = 30

ANOMALY_KINDS = ('blowup', 'noise', 'tachy', 'brady', 'afib')
N_FEATURES = 17

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

def weighted_mean_std(stats: list[tuple[int, float, float]]) -> tuple[float, float]:
    """Combine per-subject (count, mean, std) into a single mean/std, weighting
    each subject by its sample count so longer recordings count proportionally."""
    sizes = np.array([n for n, _, _ in stats], dtype=np.float64)
    means = np.array([m for _, m, _ in stats], dtype=np.float64)
    stds  = np.array([s for _, _, s in stats], dtype=np.float64)
    total = sizes.sum()
    mean = float((sizes * means).sum() / total)
    std  = float((sizes * stds).sum() / total)
    return mean, std


def extract_subject_signals(raw_dir: Path, subjects_dir: Path) -> list[int]:
    """Extract raw BVP (64 Hz) and ACC magnitude (32 Hz) per subject.

    BVP and ACC are stored raw (un-normalized, different lengths) so anomaly
    injection and load-time normalization can work from a single source. The global
    BVP mean/std (size-weighted across subjects) goes to norm-params.npy. ACC is
    stored for feature extraction only; it is not fed to any model, so it needs no
    normalization params.
    """
    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_raw_dirs = get_sorted_paths(raw_dir)

    bvp_stats: list[tuple[int, float, float]] = []
    processed = []

    for subject_raw_dir in subject_raw_dirs:
        subject_dir_name = subject_raw_dir.name
        path = subject_raw_dir / f'{subject_dir_name}.pkl'
        raw = pkl.loads(path.read_bytes(), encoding='latin1')

        wrist = raw['signal']['wrist']
        bvp = wrist['BVP'].flatten().astype(np.float32)

        acc_g = wrist['ACC'] / 64.0
        acc = np.sqrt(np.sum(acc_g ** 2, axis=1)).astype(np.float32)

        save_dir = subjects_dir / subject_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy', bvp)
        np.save(save_dir / 'acc.npy', acc)

        bvp_stats.append((len(bvp), float(bvp.mean()), float(bvp.std())))
        processed.append(subject_dir_name)

        print(f"  {subject_dir_name}: BVP {len(bvp)} samples @ {BVP_RATE} Hz, ACC {len(acc)} samples @ {ACC_RATE} Hz")

    if not processed:
        return []

    bvp_mean, bvp_std = weighted_mean_std(bvp_stats)
    np.save(subjects_dir / NORM_PARAMS_FILE,
            np.array([bvp_mean, bvp_std], dtype=np.float32))

    print(f"  Global BVP mean/std saved to {subjects_dir / NORM_PARAMS_FILE}")
    return processed


# ---------------------------------------------------------------------------
# Stage 2 — Synthetic anomalies on raw BVP
# ---------------------------------------------------------------------------

def wavy_noise(n: int, std: float) -> np.ndarray:
    """Smooth band-limited random noise over ``n`` samples, std-normalized to ``std``."""
    spacing = int(np.random.randint(8, 25))
    m = max(3, n // spacing)
    noise = np.fft.irfft(np.fft.rfft(np.random.normal(0.0, 1.0, size=m)), n)
    sd = float(noise.std())
    return noise / sd * std

def stretch_by(factor, x, y):
    m = int(round(len(x) * factor))
    return np.interp(np.linspace(0, len(x) - 1, m), x, y)

def apply_anomaly(segment: np.ndarray, kind: int, sig_std: float) -> np.ndarray:
    """Return a perturbed copy of a BVP ``segment`` for ``ANOMALY_KINDS[kind]``."""
    seg = segment.copy()
    n = len(seg)
    src = np.linspace(0, n - 1, n)

    if kind == 0:    # blowup - amplitude blow-up around the local mean
        mean = float(seg.mean())
        seg = mean + (seg - mean) * float(np.random.uniform(2.0, 4.0))

    elif kind == 1:  # noise - wavy band-limited interference burst
        seg += wavy_noise(n, sig_std * float(np.random.uniform(0.25, 0.4)))

    elif kind == 2:  # tachycardia - increased tempo by shrinking and tiling
        factor = np.random.uniform(0.5, 0.65)
        resampled = stretch_by(factor, src, seg)
        seg = np.tile(resampled, int(np.ceil(n / len(resampled))))[:n]

    elif kind == 3:  # bradycardia - decreased tempo by stretching
        factor = float(np.random.uniform(1.5, 1.65))
        resampled = stretch_by(factor, src, seg)
        seg = resampled[:n]

    else:            # afib - irregularly-irregular rhythm via a jittered monotonic warp
        n_ctrl = max(2, n // BVP_RATE)   # ~1 speed control point per second
        speed = np.interp(src, np.linspace(0, n - 1, n_ctrl),
                          np.random.uniform(0.3, 1.7, size=n_ctrl))
        warp = np.cumsum(speed)
        warp *= (n - 1) / warp[-1]       # normalize to [0, n-1], endpoints fixed
        seg = np.interp(warp, src, seg)

    return seg.astype(np.float32)


def inject_mixed(bvp: np.ndarray, anomaly_prob: float) -> tuple[np.ndarray, np.ndarray]:
    """Inject a window-aligned mix of random anomaly kinds into a raw BVP signal.

    Anomalies span whole ``BVP_WINDOW``-sample windows (no partial-overlap windows),
    so the per-window binary label maps 1:1 onto the feature/distillation grid.
    Returns (anomalous_bvp, win_labels) with win_labels of length
    ``len(bvp) // BVP_WINDOW``.
    """
    result = bvp.copy()
    n_windows = len(bvp) // BVP_WINDOW
    win_labels = np.zeros(n_windows, dtype=np.float32)
    if n_windows == 0:
        return result.astype(np.float32), win_labels

    sig_std   = float(bvp.std())
    target = int(n_windows * anomaly_prob)

    while int(win_labels.sum()) < target:
        length = random.randint(MIN_ANOMALY_WINDOWS, MAX_ANOMALY_WINDOWS)
        start  = random.randint(0, n_windows - length + 1)

        wins   = slice(start, start + length)
        if win_labels[wins].any():
            continue

        seg = slice(start * BVP_WINDOW, (start + length) * BVP_WINDOW)
        kind = int(random.randrange(len(ANOMALY_KINDS)))
        result[seg] = apply_anomaly(result[seg], kind, sig_std)

        win_labels[wins] = 1.0

    return result.astype(np.float32), win_labels


def inject_single_kind(bvp: np.ndarray, kind: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one anomaly kind to every window of a raw BVP signal by tiling
    window-aligned spans across it — a fully-anomalous per-type signal for isolated
    testing (every window is an example of ``kind``)."""
    result = bvp.copy()
    n_windows = len(bvp) // BVP_WINDOW
    if n_windows == 0:
        return result.astype(np.float32)

    sig_std   = float(bvp.std())

    w = 0
    while w < n_windows:
        length = min(int(rng.integers(MIN_ANOMALY_WINDOWS, MAX_ANOMALY_WINDOWS + 1)), n_windows - w)
        seg = slice(w * BVP_WINDOW, (w + length) * BVP_WINDOW)
        result[seg] = apply_anomaly(result[seg], kind, sig_std)
        w += length

    return result.astype(np.float32)


def create_anomalous_signals(subjects_dir: Path, anomalous_dir: Path):
    """Per-type fully-anomalous BVP for isolated testing: for each kind in
    ANOMALY_KINDS, apply it to every window of each subject's clean BVP. Layout:
    ``<anomalous_dir>/<kind>/S*/bvp.npy``. ACC is unchanged (load from clean-signals).
    """
    rng = np.random.default_rng(SEED)

    for kind, name in enumerate(ANOMALY_KINDS):
        kind_dir = anomalous_dir / name
        subject_dirs = get_sorted_paths(subjects_dir)
        for subject_dir in subject_dirs:
            sid = subject_dir.name
            bvp = np.load(subject_dir / 'bvp.npy')
            anomalous_bvp = inject_single_kind(bvp, kind, rng)
            save_dir = kind_dir / sid
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / 'bvp.npy', anomalous_bvp)
        print(f"  {name}: {len(subject_dirs)} subjects")


def create_mixed_signals(subjects_dir: Path, mixed_dir: Path):
    """Realistic ~ANOMALY_PROB mix of anomaly kinds on window-aligned spans.

    Only BVP is modified; ACC is not stored here (load from clean-signals). bvp.npy
    + per-window binary labels.npy; used for threshold-picking, distillation and
    feature-mlp training.
    """
    mixed_dir.mkdir(parents=True, exist_ok=True)

    for subject_dir in get_sorted_paths(subjects_dir):
        subject_id = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')

        mixed_bvp, win_labels = inject_mixed(bvp, ANOMALY_PROB)

        save_dir = mixed_dir / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy',    mixed_bvp)
        np.save(save_dir / 'labels.npy', win_labels)

        print(f"  {subject_id}: {len(win_labels)} windows, {win_labels.mean():.1%} anomalous")


# ---------------------------------------------------------------------------
# Stage 3 — Feature dataset
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


def build_feature_dataset(signal_dir: Path, subjects_dir: Path, feature_dir: Path):
    """Window BVP and raw ACC into non-overlapping 8-second windows and extract features.

    BVP comes from ``signal_dir`` (raw, un-normalized): mixed-signals for the training
    set, or clean-signals for the anomaly-free export set. ACC comes from
    ``subjects_dir`` (clean-signals; raw magnitude, 32 Hz) — the anomalies are injected
    into BVP only. Per-window labels are read from ``signal_dir/S*/labels.npy``
    when present (mixed anomalies are window-aligned, so each window is fully clean or
    fully anomalous); the clean signals carry no labels file, so every window is normal
    (label 0). Features are stored per subject under S*/ raw (un-normalized), mirroring
    what the device echoes; the global z-score stats are saved at the top level
    (feature_stats.npy) and baked into the model as its z-score constants.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)

    per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("Building feature dataset from:", end="")
    for subject_dir in get_sorted_paths(signal_dir):
        subject_id = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')          # mixed-anomaly or clean BVP
        acc = np.load(subjects_dir / subject_id / 'acc.npy')

        n_windows = min(
            len(bvp) // BVP_WINDOW,
            (len(acc) - ACC_WINDOW) // ACC_WINDOW + 1,
        )

        label_path = subject_dir / 'labels.npy'
        win_lbl = (np.load(label_path) if label_path.exists()
                   else np.zeros(max(0, n_windows), dtype=np.float32))

        features: list[np.ndarray] = []
        labels:   list[float]      = []
        for i in range(max(0, n_windows)):
            bvp_start = i * BVP_WINDOW
            acc_start = i * ACC_WINDOW

            bvp_win = bvp[bvp_start : bvp_start + BVP_WINDOW]
            acc_win = acc[acc_start : acc_start + ACC_WINDOW]

            features.append(extract_features(bvp_win, acc_win))
            labels.append(float(win_lbl[i]))

        per_subject[subject_id] = (
            np.stack(features),
            np.asarray(labels, dtype=np.float32).reshape(-1, 1),
        )
        print(f" {subject_id}", end="", flush=True)
    print()

    all_x = np.concatenate([x for x, _ in per_subject.values()])
    mean = all_x.mean(axis=0)
    std  = all_x.std(axis=0) + 1e-8

    total = anomalous = 0
    for subject_id, (x, y) in per_subject.items():
        save_dir = feature_dir / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'features.npy', x.astype(np.float32))
        np.save(save_dir / 'labels.npy',   y)
        total += len(y)
        anomalous += int(y.sum())

    np.save(feature_dir / FEATURE_STATS_FILE, np.stack([mean, std]).astype(np.float32))
    print(f"Saved {total} windows ({anomalous} anomalous) to {feature_dir}")

