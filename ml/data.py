"""Dataset processing and loading for the PPG-DaLiA anomaly-detection pipeline.

Owns everything between the raw download and the tensors the trainers consume:
the per-stage build functions (raw extraction, synthetic-anomaly injection,
feature extraction), the windowing/normalization helpers used at load time, and
the shared constants. ``scripts/get_dataset.py`` is a thin CLI that downloads the
archive and sequences these stages; the model trainers and ``distill_labels.py``
call the load helpers here.
"""

import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

RAW_SUBDIR = 'PPG_FieldStudy'
SUBJECTS_SUBDIR = 'subject-signals'
ANOMALOUS_SUBDIR = 'anomalous-signals'      # per-type fully-anomalous BVP: <kind>/S*/
MIXED_SUBDIR = 'mixed-signals'              # realistic ~50% mix: S*/ (bvp + binary labels)
MIXED_FEATURE_SUBDIR = 'mixed-features'     # features windowed from mixed-signals
NORM_PARAMS_FILE = 'norm-params.npy'
STATIC_NORM_PARAMS_FILE = 'static_norm_params.npy'
FEATURE_STATS_FILE = 'feature_stats.npy'

BVP_RATE = 64
ACC_RATE = 32
WINDOW_SECONDS = 8
BVP_WINDOW = BVP_RATE * WINDOW_SECONDS    # 512 samples
ACC_WINDOW = ACC_RATE * WINDOW_SECONDS    # 256 samples
ANOMALY_PROB = 0.5
MIN_ANOMALY_WINDOWS = 8
MAX_ANOMALY_WINDOWS = 30
# Synthetic anomaly kinds (apply_anomaly index). Signal-integrity artifacts
# (spike/blowup/noise) + rhythm anomalies (timewarp = local tachy/brady, afib =
# irregularly-irregular rhythm). Flatline and baseline wander were dropped: a
# flatline is below the AE's reconstruction error floor (catch it with a signal-
# quality gate instead) and wander is physiological (already in the clean signal).
ANOMALY_KINDS = ('spike', 'blowup', 'noise', 'timewarp', 'afib')
FEATURE_SEED = 1234
N_FEATURES = 17
EPS = 1e-8


class DatasetUnavailibleError(FileNotFoundError):
    def __init__(self, topic: str, data_dir: str | Path):
        self.message = f"{topic} dataset not found at {data_dir}. Run scripts/get_dataset.py first."
        super().__init__(self.message)


def user_description_vector(quest: dict) -> np.ndarray:
    """Raw 6-dim demographics vector. Standardized globally in Stage 1."""
    gender_raw = quest.get('Gender', 'm').strip().lower()
    gender = 1.0 if gender_raw == 'f' else 0.0
    return np.array([
        gender,
        float(quest.get('AGE', 30)),
        float(quest.get('HEIGHT', 150)),
        float(quest.get('WEIGHT', 70)),
        float(quest.get('SKIN', 3)),
        float(quest.get('SPORT', 3)),
    ], dtype=np.float32)


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
    injection and load-time normalization can work from a single source. Global
    BVP/ACC mean/std (size-weighted across subjects) go to norm-params.npy; the
    6-dim demographics vector is z-scored across subjects (params in
    static_norm_params.npy) and stored already-normalized as static.npy.
    """
    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = sorted(
        int(p.name[1:]) for p in raw_dir.glob('S*')
        if p.is_dir() and p.name[1:].isdigit()
    )

    bvp_stats: list[tuple[int, float, float]] = []
    acc_stats: list[tuple[int, float, float]] = []
    statics: dict[int, np.ndarray] = {}
    processed = []

    for subject_id in subject_ids:
        path = raw_dir / f'S{subject_id}' / f'S{subject_id}.pkl'
        raw = pkl.loads(path.read_bytes(), encoding='latin1')

        wrist = raw['signal']['wrist']
        bvp = wrist['BVP'].flatten().astype(np.float32)

        acc_g = wrist['ACC'] / 64.0
        acc = np.sqrt(np.sum(acc_g ** 2, axis=1)).astype(np.float32)

        save_dir = subjects_dir / f'S{subject_id}'
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy', bvp)
        np.save(save_dir / 'acc.npy', acc)

        statics[subject_id] = user_description_vector(raw['questionnaire'])
        bvp_stats.append((len(bvp), float(bvp.mean()), float(bvp.std())))
        acc_stats.append((len(acc), float(acc.mean()), float(acc.std())))

        processed.append(subject_id)
        print(f"  S{subject_id}: BVP {len(bvp)} samples @ {BVP_RATE} Hz, ACC {len(acc)} samples @ {ACC_RATE} Hz")

    if not processed:
        return []

    bvp_mean, bvp_std = weighted_mean_std(bvp_stats)
    acc_mean, acc_std = weighted_mean_std(acc_stats)
    np.save(subjects_dir / NORM_PARAMS_FILE,
            np.array([[bvp_mean, bvp_std], [acc_mean, acc_std]], dtype=np.float32))

    all_static = np.stack([statics[sid] for sid in processed])
    static_mean = all_static.mean(axis=0).astype(np.float32)
    static_std = (all_static.std(axis=0) + EPS).astype(np.float32)
    np.save(subjects_dir / STATIC_NORM_PARAMS_FILE,
            np.stack([static_mean, static_std]).astype(np.float32))
    for subject_id in processed:
        norm_static = ((statics[subject_id] - static_mean) / static_std).astype(np.float32)
        np.save(subjects_dir / f'S{subject_id}' / 'static.npy', norm_static)

    print(f"  Global BVP/ACC mean/std saved to {subjects_dir / NORM_PARAMS_FILE}")
    print(f"  Static mean/std saved to {subjects_dir / STATIC_NORM_PARAMS_FILE}")
    return processed


# ---------------------------------------------------------------------------
# Stage 2 — Synthetic anomalies on raw BVP
# ---------------------------------------------------------------------------

def apply_anomaly(segment: np.ndarray, kind: int, rng: np.random.Generator,
                  sig_range: float, sig_std: float) -> np.ndarray:
    """Return a perturbed copy of a BVP ``segment`` for ``ANOMALY_KINDS[kind]``.

    Perturbations are scaled by the signal's own range/std so they are equally
    disruptive at any sensor output range. The rhythm anomalies (timewarp/afib) are
    monotonic time-warps via ``np.interp`` — they need no beat detection and stay
    within the segment.
    """
    seg = segment.astype(np.float32).copy()
    n = len(seg)
    src = np.linspace(0, n - 1, n)

    if kind == 0:    # spike — transient DC offset over the span
        scale = sig_range * float(rng.uniform(0.3, 0.8))
        seg += scale * float(rng.choice([-1.0, 1.0]))
    elif kind == 1:  # blowup — amplitude blow-up around the local mean
        mean = float(seg.mean())
        seg = mean + (seg - mean) * float(rng.uniform(2.0, 4.0))
    elif kind == 2:  # noise — additive noise burst
        seg += rng.normal(0.0, sig_std * 0.5, size=n).astype(np.float32)
    elif kind == 3:  # timewarp — uniform tempo change (tachy/brady): resample + refit
        factor = float(rng.uniform(1.4, 1.8)) if rng.random() < 0.5 else float(rng.uniform(0.55, 0.7))
        m = max(2, int(round(n / factor)))
        resampled = np.interp(np.linspace(0, n - 1, m), src, seg).astype(np.float32)
        seg = (resampled[:n] if m >= n
               else np.tile(resampled, int(np.ceil(n / m)))[:n]).astype(np.float32)
    else:            # afib — irregularly-irregular rhythm via a jittered monotonic warp
        n_ctrl = max(2, n // BVP_RATE)                  # ~1 speed control point per second
        speed = np.interp(src, np.linspace(0, n - 1, n_ctrl),
                          rng.uniform(0.3, 1.7, size=n_ctrl))
        warp = np.cumsum(speed)
        warp *= (n - 1) / warp[-1]                      # normalize to [0, n-1], endpoints fixed
        seg = np.interp(warp, src, seg).astype(np.float32)

    return seg.astype(np.float32)


def inject_mixed(bvp: np.ndarray, rng: np.random.Generator,
                 anomaly_prob: float) -> tuple[np.ndarray, np.ndarray]:
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

    sig_range = float(bvp.max() - bvp.min())
    sig_std   = float(bvp.std())
    min_w  = min(MIN_ANOMALY_WINDOWS, n_windows)
    max_w  = max(min_w, min(MAX_ANOMALY_WINDOWS, n_windows))
    target = int(n_windows * anomaly_prob)

    attempts = 0
    while int(win_labels.sum()) < target and attempts < 10_000:
        attempts += 1
        length = int(rng.integers(min_w, max_w + 1))
        start  = int(rng.integers(0, max(1, n_windows - length + 1)))
        wins   = slice(start, start + length)
        if win_labels[wins].any():
            continue

        seg = slice(start * BVP_WINDOW, (start + length) * BVP_WINDOW)
        kind = int(rng.integers(0, len(ANOMALY_KINDS)))
        result[seg] = apply_anomaly(result[seg], kind, rng, sig_range, sig_std)
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

    sig_range = float(bvp.max() - bvp.min())
    sig_std   = float(bvp.std())
    min_w = min(MIN_ANOMALY_WINDOWS, n_windows)
    max_w = max(min_w, min(MAX_ANOMALY_WINDOWS, n_windows))

    w = 0
    while w < n_windows:
        length = min(int(rng.integers(min_w, max_w + 1)), n_windows - w)
        seg = slice(w * BVP_WINDOW, (w + length) * BVP_WINDOW)
        result[seg] = apply_anomaly(result[seg], kind, rng, sig_range, sig_std)
        w += length

    return result.astype(np.float32)


def create_anomalous_signals(subjects_dir: Path, anomalous_dir: Path):
    """Per-type fully-anomalous BVP for isolated testing: for each kind in
    ANOMALY_KINDS, apply it to every window of each subject's clean BVP. Layout:
    ``<anomalous_dir>/<kind>/S*/bvp.npy``. ACC is unchanged (load from subject-signals).
    """
    rng = np.random.default_rng(FEATURE_SEED)

    for kind, name in enumerate(ANOMALY_KINDS):
        kind_dir = anomalous_dir / name
        subject_dirs = sorted(subjects_dir.glob('S*'))
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

    Only BVP is modified; ACC is not stored here (load from subject-signals). bvp.npy
    + per-window binary labels.npy; used for threshold-picking, distillation and
    feature-mlp training.
    """
    mixed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FEATURE_SEED)

    for subject_dir in sorted(subjects_dir.glob('S*')):
        subject_id = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')

        mixed_bvp, win_labels = inject_mixed(bvp, rng, ANOMALY_PROB)

        save_dir = mixed_dir / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy',    mixed_bvp)
        np.save(save_dir / 'labels.npy', win_labels)

        print(f"  {subject_id}: {len(win_labels)} windows, {win_labels.mean():.1%} anomalous")


# ---------------------------------------------------------------------------
# Stage 3 — Feature dataset
# ---------------------------------------------------------------------------

def extract_features(bvp_window: np.ndarray, acc_window: np.ndarray) -> np.ndarray:
    """17-feature vector from an 8-second BVP window (512 samples) and ACC window (256 samples).

    Must stay in sync with firmware/main/ml/features.c.
    """
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


def build_feature_dataset(mixed_dir: Path, subjects_dir: Path, feature_dir: Path):
    """Window mixed-anomaly BVP and raw ACC into non-overlapping 8-second windows and
    extract features.

    BVP comes from mixed-signals (raw with the anomaly mix, un-normalized).
    ACC comes from subject-signals (raw magnitude, 32 Hz).
    Per-window labels are taken straight from mixed-signals (anomalies are
    window-aligned, so each window is fully clean or fully anomalous).
    Features are stored per subject under S*/; standardization (z-score) stats
    are still computed globally and saved at the top level for on-device use.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)

    per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("Building feature dataset from:", end="")
    for subject_dir in sorted(mixed_dir.glob('S*')):
        subject_id = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')
        win_lbl = np.load(subject_dir / 'labels.npy')   # per-window
        acc = np.load(subjects_dir / subject_id / 'acc.npy')

        n_windows = min(
            len(win_lbl),
            (len(acc) - ACC_WINDOW) // ACC_WINDOW + 1,
        )

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
        np.save(save_dir / 'features.npy', ((x - mean) / std).astype(np.float32))
        np.save(save_dir / 'labels.npy',   y)
        total += len(y)
        anomalous += int(y.sum())

    np.save(feature_dir / FEATURE_STATS_FILE, np.stack([mean, std]).astype(np.float32))
    print(f"Saved {total} windows ({anomalous} anomalous) to {feature_dir}")


# ---------------------------------------------------------------------------
# Load-time helpers (windowing + normalization)
# ---------------------------------------------------------------------------

def window_signal(signal: np.ndarray, window_size: int, shift: int):
    """Window a ``(T, n_signals)`` array into ``(window_size, n_signals)`` frames.

    Returns the windowed dataset and its (asserted) cardinality.
    """
    count = (len(signal) - window_size) // shift + 1
    ds = (tf.data.Dataset.from_tensor_slices(signal)
          .window(size=window_size, shift=shift, drop_remainder=True)
          .flat_map(lambda w: w.batch(window_size, drop_remainder=True))
          .apply(tf.data.experimental.assert_cardinality(count)))
    return ds, count


def load_norm_params(subjects_dir: Path) -> tuple[float, float, float, float]:
    """Read the global BVP/ACC (mean, std) saved by Stage 1, guarding against a
    zero std. Returns (bvp_mean, bvp_std, acc_mean, acc_std)."""
    params = np.load(subjects_dir / NORM_PARAMS_FILE)
    bvp_mean, bvp_std = float(params[0][0]), float(params[0][1])
    acc_mean, acc_std = float(params[1][0]), float(params[1][1])
    return bvp_mean, bvp_std + EPS, acc_mean, acc_std + EPS


def norm_stats(subjects_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(mean, std)`` for the stacked ``[BVP, ACC]`` signal."""
    bvp_mean, bvp_std, acc_mean, acc_std = load_norm_params(subjects_dir)
    mean = np.array([bvp_mean, acc_mean], dtype=np.float32)
    std  = np.array([bvp_std, acc_std], dtype=np.float32)
    return mean, std


def _interp_acc(acc: np.ndarray, target_len: int) -> np.ndarray:
    """Resample ACC (32 Hz) to the BVP sample count (64 Hz)."""
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(acc)),
        acc,
    ).astype(np.float32)


def stacked_signal(subjects_dir: Path, sid: str,
                   anomalous_dir: Path | None = None) -> np.ndarray:
    """Raw, un-normalized ``(T, 2)`` ``[BVP(64 Hz), ACC(interp to 64 Hz)]`` for a
    subject. BVP is read from ``anomalous_dir`` when given, else subject-signals;
    ACC always comes from subject-signals."""
    bvp_src = anomalous_dir if anomalous_dir is not None else subjects_dir
    bvp = np.load(bvp_src / sid / 'bvp.npy').astype(np.float32)
    acc = np.load(subjects_dir / sid / 'acc.npy').astype(np.float32)
    acc = _interp_acc(acc, len(bvp))
    return np.stack([bvp, acc], axis=-1).astype(np.float32)


def normalize(signal: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((signal - mean) / std).astype(np.float32)


def windowed_normalized(subjects_dir: Path, sid: str, window_size: int, shift: int,
                        anomalous_dir: Path | None = None) -> tuple[tf.data.Dataset, int]:
    """Windowed, z-score-normalized ``[BVP, ACC]`` frames for one subject.

    Normalization is applied as a ``map`` over the windows (rather than stored on
    disk), mirroring the on-device path where raw samples are normalized as they
    are read."""
    raw = stacked_signal(subjects_dir, sid, anomalous_dir)
    mean, std = norm_stats(subjects_dir)
    ds, count = window_signal(raw, window_size, shift)
    mean_t, std_t = tf.constant(mean), tf.constant(std)
    return ds.map(lambda w: (w - mean_t) / std_t), count
