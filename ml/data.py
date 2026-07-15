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

from common.config import SEED

RAW_SUBDIR = 'PPG_FieldStudy'
CLEAN_SUBDIR = 'clean-signals'
ANOMALOUS_SUBDIR = 'anomalous-signals'      # per-type fully-anomalous BVP: <kind>/S*/
MIXED_SUBDIR = 'mixed-signals'              # realistic ~50% mix: S*/ (bvp + binary labels)
MIXED_FEATURE_SUBDIR = 'mixed-features'     # features windowed from mixed-signals
CLEAN_FEATURE_SUBDIR = 'clean-features'     # features windowed from clean-signals (all-normal)
NORM_PARAMS_FILE = 'norm-params.npy'
STATIC_NORM_PARAMS_FILE = 'static_norm_params.npy'
CONTEXT_NORM_PARAMS_FILE = 'context_norm_params.npy'
FEATURE_STATS_FILE = 'feature_stats.npy'
CONTEXT_FILE = 'context.npy'

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
N_FEATURES = 17
EPS = 1e-8

# Conditioning vector for the autoencoders: z-scored demographics (static) + a causal
# activity context (trailing-CONTEXT_SECONDS mean/std of the normalized ACC magnitude).
# The context lets the decoder expect activity-appropriate rhythm (a fast pulse is normal
# under high ACC, anomalous at rest). Warm-up (<CONTEXT_SECONDS of history) expands over
# whatever is available — all subjects start at rest, so that's the at-rest average.
CONTEXT_SECONDS = 120
N_STATIC = 6
N_CONTEXT = 2
N_COND = N_STATIC + N_CONTEXT


class DatasetUnavailibleError(FileNotFoundError):
    def __init__(self, data_dir: str | Path):
        self.message = f"Dataset not found at {data_dir}. Run scripts/get_dataset.py first."
        super().__init__(self.message)


def get_sorted_paths(dataset_dir: Path) -> list[Path]:
    dir_list = [d for d in dataset_dir.glob('S*') if d.is_dir() and d.name[1:].isdigit()]
    return sorted(dir_list, key=lambda d: int(d.name[1:]))

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

    subject_raw_dirs = get_sorted_paths(raw_dir)

    bvp_stats: list[tuple[int, float, float]] = []
    acc_stats: list[tuple[int, float, float]] = []
    statics: dict[str, np.ndarray] = {}
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
        acc_stats.append((len(acc), float(acc.mean()), float(acc.std())))

        statics[subject_dir_name] = user_description_vector(raw['questionnaire'])
        processed.append(subject_dir_name)

        print(f"  {subject_dir_name}: BVP {len(bvp)} samples @ {BVP_RATE} Hz, ACC {len(acc)} samples @ {ACC_RATE} Hz")

    if not processed:
        return []

    bvp_mean, bvp_std = weighted_mean_std(bvp_stats)
    acc_mean, acc_std = weighted_mean_std(acc_stats)
    np.save(subjects_dir / NORM_PARAMS_FILE,
            np.array([[bvp_mean, bvp_std], [acc_mean, acc_std]], dtype=np.float32))

    # Context normalization params: the activity context is the trailing mean/std of
    # the raw ACC magnitude, so normalizing it reproduces the old "normalize ACC, then
    # take trailing stats" exactly — the mean dim shifts by acc_mean and both dims
    # scale by acc_std. Stored separately (redundant with norm-params) so the context
    # path has self-contained params to apply at load time.
    ctx_mean = np.array([acc_mean, 0.0], dtype=np.float32)
    ctx_std = np.array([acc_std + EPS, acc_std + EPS], dtype=np.float32)
    np.save(subjects_dir / CONTEXT_NORM_PARAMS_FILE,
            np.stack([ctx_mean, ctx_std]).astype(np.float32))

    all_static = np.stack([statics[d] for d in processed])
    static_mean = all_static.mean(axis=0).astype(np.float32)
    static_std = (all_static.std(axis=0) + EPS).astype(np.float32)
    np.save(subjects_dir / STATIC_NORM_PARAMS_FILE,
            np.stack([static_mean, static_std]).astype(np.float32))
    for subject_id in processed:
        norm_static = ((statics[subject_id] - static_mean) / static_std).astype(np.float32)
        np.save(subjects_dir / subject_id / 'static.npy', norm_static)

    print(f"  Global BVP/ACC mean/std saved to {subjects_dir / NORM_PARAMS_FILE}")
    print(f"  Static mean/std saved to {subjects_dir / STATIC_NORM_PARAMS_FILE}")
    print(f"  Context norm params saved to {subjects_dir / CONTEXT_NORM_PARAMS_FILE}")
    return processed


def build_context_pass(subjects_dir: Path):
    """Precompute each subject's per-window activity context on the non-overlapping
    8-second grid -> ``S*/context.npy`` (raw, un-normalized; consumers normalize at
    load with ``context_norm_params.npy``).

    This is the precomputed counterpart of the at-load context in ``window_cond_vectors``
    for the no-overlap case: it's consumed by ``conditional_windows`` and exported by
    ``scripts/export_subject_data.py`` so the phone can store context per window. The
    shifted ``windowed_conditional`` path does not use it (it samples context at
    arbitrary window ends and so recomputes)."""
    # Self-heal context norm params for datasets whose Stage 1 predates them.
    if not (subjects_dir / CONTEXT_NORM_PARAMS_FILE).exists():
        _, _, acc_mean, acc_std = load_norm_params(subjects_dir)   # acc_std already + EPS
        ctx_mean = np.array([acc_mean, 0.0], dtype=np.float32)
        ctx_std = np.array([acc_std, acc_std], dtype=np.float32)
        np.save(subjects_dir / CONTEXT_NORM_PARAMS_FILE,
                np.stack([ctx_mean, ctx_std]).astype(np.float32))

    for subject_dir in get_sorted_paths(subjects_dir):
        sid = subject_dir.name
        raw_acc = stacked_signal(subjects_dir, sid)[:, 1]
        count = max(0, (len(raw_acc) - BVP_WINDOW) // BVP_WINDOW + 1)
        ctx_mean, ctx_std = causal_rolling_mean_std(raw_acc, CONTEXT_SECONDS * BVP_RATE)
        ends = np.clip(np.arange(count) * BVP_WINDOW + BVP_WINDOW - 1, 0, len(raw_acc) - 1)
        ctx = np.stack([ctx_mean[ends], ctx_std[ends]], axis=1).astype(np.float32)
        np.save(subject_dir / CONTEXT_FILE, ctx)
        print(f"  {sid}: {count} context windows")


# ---------------------------------------------------------------------------
# Stage 2 — Synthetic anomalies on raw BVP
# ---------------------------------------------------------------------------

def wavy_noise(n: int, rng: np.random.Generator, std: float) -> np.ndarray:
    """Smooth band-limited random noise over ``n`` samples, std-normalized to ``std``.

    Random values on control points every ~0.12–0.38 s are interpolated to fill the
    rest, so the result is wavy and below the PPG band rather than per-sample hiss.
    Control points are evenly spaced (filled by FFT/trigonometric interpolation) or
    randomly placed (smoothed linear interpolation), chosen per call for variety.
    """
    spacing = int(rng.integers(8, 25))
    m = max(3, n // spacing)
    if rng.random() < 0.5:                                   # regular grid → trig interp
        noise = np.fft.irfft(np.fft.rfft(rng.normal(0.0, 1.0, size=m)), n)
    else:                                                    # random points → smoothed linear
        pos = np.concatenate((
            [0],
            np.sort(rng.choice(np.arange(1, n - 1), size=min(m, n - 2), replace=False)),
            [n - 1]))
        noise = np.interp(np.arange(n), pos, rng.normal(0.0, 1.0, size=len(pos)))
        kernel = np.hanning(max(3, spacing))
        noise = np.convolve(noise, kernel / kernel.sum(), mode='same')
    sd = float(noise.std())
    return (noise / sd * std if sd > 0 else noise).astype(np.float32)


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

    if kind == 0:    # spike — baseline step (sustained DC offset over the span)
        scale = sig_range * float(rng.uniform(0.05, 0.1))
        seg += scale * float(rng.choice([-1.0, 1.0]))
    elif kind == 1:  # blowup — amplitude blow-up around the local mean
        mean = float(seg.mean())
        seg = mean + (seg - mean) * float(rng.uniform(2.0, 4.0))
    elif kind == 2:  # noise — wavy band-limited interference burst
        seg += wavy_noise(n, rng, sig_std * float(rng.uniform(0.25, 0.4)))
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
    rng = np.random.default_rng(SEED)

    for subject_dir in get_sorted_paths(subjects_dir):
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
    set, or clean-signals for the anomaly-free export set. ACC comes from clean-signals
    (raw magnitude, 32 Hz). Per-window labels are read from ``signal_dir/S*/labels.npy``
    when present (mixed anomalies are window-aligned, so each window is fully clean or
    fully anomalous); the clean signals carry no labels file, so every window is normal
    (label 0). Features are stored per subject under S*/ raw (un-normalized), mirroring
    what the device echoes; the global z-score stats are saved at the top level
    (feature_stats.npy) and applied at load time (see FeatureMLPTrainer.subject_dataset).
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


# ---------------------------------------------------------------------------
# Load-time helpers (windowing + normalization)
# ---------------------------------------------------------------------------

def window_signal(
        signal: np.ndarray, window_size: int, shift: int
    ) -> tf.data.Dataset:
    """Window a ``(T, n_signals)`` array into ``(window_size, n_signals)`` frames.

    Returns the windowed dataset and its (asserted) cardinality.
    """
    count = (len(signal) - window_size) // shift + 1
    ds = (tf.data.Dataset.from_tensor_slices(signal)
          .window(size=window_size, shift=shift, drop_remainder=True)
          .flat_map(lambda w: w.batch(window_size, drop_remainder=True))
          .apply(tf.data.experimental.assert_cardinality(count)))
    return ds


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


def load_feature_stats(feature_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ``(mean, std)`` saved by Stage 4, applied to the raw on-disk
    features at load time."""
    stats = np.load(feature_dir / FEATURE_STATS_FILE)
    return stats[0].astype(np.float32), stats[1].astype(np.float32)


def load_context_norm_params(subjects_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-context-dim ``(mean, std)`` saved by Stage 1, applied to the raw activity
    context (trailing mean/std of the raw ACC) at load time."""
    params = np.load(subjects_dir / CONTEXT_NORM_PARAMS_FILE)
    return params[0].astype(np.float32), params[1].astype(np.float32)


def load_static_norm_params(subjects_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-demographic ``(mean, std)`` saved by Stage 1, baked into the model so it
    z-scores the raw 6-d static vector (part of the cond) in its own signatures."""
    params = np.load(subjects_dir / STATIC_NORM_PARAMS_FILE)
    return params[0].astype(np.float32), params[1].astype(np.float32)


def load_static_raw(subjects_dir: Path, sid: str) -> np.ndarray:
    """Raw demographics vector. ``static.npy`` is stored z-scored (Stage 1); since the
    model now owns cond normalization, de-normalize it back to raw for the load path."""
    norm = np.load(subjects_dir / sid / 'static.npy').astype(np.float32)
    mean, std = load_static_norm_params(subjects_dir)
    return (norm * std + mean).astype(np.float32)


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
    subject. BVP is read from ``anomalous_dir`` when given, else clean-signals;
    ACC always comes from clean-signals."""
    bvp_src = anomalous_dir if anomalous_dir is not None else subjects_dir
    bvp = np.load(bvp_src / sid / 'bvp.npy').astype(np.float32)
    acc = np.load(subjects_dir / sid / 'acc.npy').astype(np.float32)
    acc = _interp_acc(acc, len(bvp))
    return np.stack([bvp, acc], axis=-1).astype(np.float32)


def causal_rolling_mean_std(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample trailing mean/std of ``x`` over the last ``win`` samples, expanding
    over whatever history exists for the first ``win`` samples (causal, no look-ahead)."""
    csum  = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    csum2 = np.concatenate([[0.0], np.cumsum(np.square(x, dtype=np.float64))])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - win)
    count = idx - lo
    mean = (csum[idx] - csum[lo]) / count
    var  = np.maximum((csum2[idx] - csum2[lo]) / count - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def window_cond_vectors(subjects_dir: Path, sid: str, raw_acc: np.ndarray,
                        window_size: int, shift: int, count: int) -> np.ndarray:
    """Per-window *raw* conditioning vectors ``(count, N_COND)``: the subject's
    demographics repeated, concatenated with the causal activity context sampled at
    each window's last sample.

    Both parts are left un-normalized — the model z-scores the whole cond vector on its
    own. The context is the trailing mean/std of the raw ACC magnitude; the same values
    the on-device pipeline computes and feeds raw."""
    static = load_static_raw(subjects_dir, sid)                             # (N_STATIC,) raw
    ctx_mean, ctx_std = causal_rolling_mean_std(raw_acc, CONTEXT_SECONDS * BVP_RATE)
    ends = np.clip(np.arange(count) * shift + window_size - 1, 0, len(raw_acc) - 1)
    ctx_raw = np.stack([ctx_mean[ends], ctx_std[ends]], axis=1)              # (count, N_CONTEXT)
    static_rep = np.broadcast_to(static, (count, len(static)))
    return np.concatenate([static_rep, ctx_raw], axis=1).astype(np.float32)


def windowed_conditional(subjects_dir: Path, sid: str, window_size: int, shift: int,
                         anomalous_dir: Path | None = None) -> tf.data.Dataset:
    """Windowed *raw* ``[BVP, ACC]`` frames paired with their per-window raw
    conditioning vector, for one subject.

    Nothing is normalized here — the model z-scores signal and cond in its own
    signatures. Yields ``(signal_window, cond)`` tuples."""
    raw = stacked_signal(subjects_dir, sid, anomalous_dir)
    ds = window_signal(raw, window_size, shift)
    cond = window_cond_vectors(subjects_dir, sid, raw[:, 1], window_size, shift, len(ds))
    cond_ds = tf.data.Dataset.from_tensor_slices(cond)
    return tf.data.Dataset.zip(ds, cond_ds)

def combine_datasets(datasets: list[tf.data.Dataset]) -> tf.data.Dataset:
    """Merge per-subject datasets."""
    count = sum([len(ds) for ds in datasets])

    return (tf.data.Dataset
            .sample_from_datasets(datasets, rerandomize_each_iteration=False)
            .apply(tf.data.experimental.assert_cardinality(count)))


def pool_datasets(datasets: list[tf.data.Dataset]) -> tf.data.Dataset:
    """Merge per-subject datasets into the single stream a centralized loop trains on,
    shuffled so gradient steps are not ordered by subject."""
    pooled = combine_datasets(datasets)
    return pooled.shuffle(len(pooled), reshuffle_each_iteration=False)


def conditional_windows(subjects_dir: Path, sid: str, window_size: int,
                        anomalous_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw ``(T, 2)`` signal + per-window raw cond ``(n_windows, N_COND)`` for
    non-overlapping windows — for distill_calibrate / distill_labels, which slice windows
    manually rather than via tf.data. The model normalizes both on eval.

    Reads the precomputed raw context (``context.npy``, on the same no-overlap grid)."""
    raw = stacked_signal(subjects_dir, sid, anomalous_dir)
    ctx_raw = np.load(subjects_dir / sid / CONTEXT_FILE).astype(np.float32)
    count = min(len(ctx_raw), max(0, (len(raw) - window_size) // window_size + 1))
    static = load_static_raw(subjects_dir, sid)
    static_rep = np.broadcast_to(static, (count, len(static)))
    cond = np.concatenate([static_rep, ctx_raw[:count]], axis=1).astype(np.float32)
    return raw, cond
