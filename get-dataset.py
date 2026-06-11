import argparse
import tempfile
import urllib.request
import zipfile
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASETS_DIR = Path('datasets')
RAW_SUBDIR = 'PPG_FieldStudy'
SUBJECTS_SUBDIR = 'subject-signals'
NORMALIZED_SUBDIR = 'normalized-signals'
ANOMALOUS_SUBDIR = 'anomalous-signals'
FEATURE_SUBDIR = 'feature-anomaly'
PARAMS_FILE = 'params.csv'

DATASET_URL = 'https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip'

BVP_RATE = 64
ACC_RATE = 32
WINDOW_SECONDS = 8
SHIFT_SECONDS = 8
BVP_WINDOW = BVP_RATE * WINDOW_SECONDS    # 512
ACC_WINDOW = ACC_RATE * WINDOW_SECONDS    # 256
BVP_SHIFT  = BVP_RATE * SHIFT_SECONDS     # 192
ACC_SHIFT  = ACC_RATE * SHIFT_SECONDS     # 96
CONTEXT_WINDOW_S = 120                    # 2 minutes
ANOMALY_PROB = 0.5
FEATURE_SEED = 1234
N_FEATURES = 17


def download_dataset(datasets_dir: Path, raw_dir: Path):
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

def user_description_vector(quest: dict) -> np.ndarray:
    gender_raw = quest.get('Gender', 'm').strip().lower()
    gender_norm = 1.0 if gender_raw == 'f' else 0.0
    age_norm    = (quest.get('AGE',     30) -  20) / 20
    height_norm = (quest.get('HEIGHT', 150) - 100) / 100
    weight_norm = (quest.get('WEIGHT',  70) -  40) / 110
    skin_norm   = (quest.get('SKIN',     3) -   1) / 5
    sport_norm  = (quest.get('SPORT',    3) -   1) / 6
    return np.array([
        gender_norm, age_norm, height_norm, weight_norm, skin_norm, sport_norm,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 1 — Extract raw signals
# ---------------------------------------------------------------------------

def extract_subject_signals(raw_dir: Path, subjects_dir: Path) -> list[int]:
    """Extract raw BVP (64 Hz) and ACC magnitude (32 Hz) per subject.

    BVP and ACC are stored in separate files because they have different lengths.
    Global min/max bounds are saved to params.csv for use in normalization.
    """
    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = sorted(
        int(p.name[1:]) for p in raw_dir.glob('S*')
        if p.is_dir() and p.name[1:].isdigit()
    )

    bvp_min = bvp_max = acc_min = acc_max = None
    processed = []

    for subject_id in subject_ids:
        path = raw_dir / f'S{subject_id}' / f'S{subject_id}.pkl'
        raw = pkl.loads(path.read_bytes(), encoding='latin1')

        wrist = raw['signal']['wrist']
        bvp = wrist['BVP'].flatten().astype(np.float32)

        acc_g = wrist['ACC'] / 64.0
        acc = np.sqrt(np.sum(acc_g ** 2, axis=1)).astype(np.float32)

        static = user_description_vector(raw['questionnaire'])

        save_dir = subjects_dir / f'S{subject_id}'
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy', bvp)
        np.save(save_dir / 'acc_mag.npy', acc)
        np.save(save_dir / 'static.npy', static)

        bvp_min = bvp.min() if bvp_min is None else min(bvp_min, float(bvp.min()))
        bvp_max = bvp.max() if bvp_max is None else max(bvp_max, float(bvp.max()))
        acc_min = acc.min() if acc_min is None else min(acc_min, float(acc.min()))
        acc_max = acc.max() if acc_max is None else max(acc_max, float(acc.max()))

        processed.append(subject_id)
        print(f"  S{subject_id}: BVP {len(bvp)} samples @ {BVP_RATE} Hz, ACC {len(acc)} samples @ {ACC_RATE} Hz")

    if not processed:
        return []

    pd.DataFrame({
        'channel': ['BVP', 'ACC_MAG'],
        'min': [bvp_min, acc_min],
        'max': [bvp_max, acc_max],
    }).to_csv(subjects_dir / PARAMS_FILE, index=False)

    print(f"  Global bounds saved to {subjects_dir / PARAMS_FILE}")
    return processed


# ---------------------------------------------------------------------------
# Stage 2 — Normalize and compute activity context
# ---------------------------------------------------------------------------

def normalize_signals(subjects_dir: Path, normalized_dir: Path):
    """Normalize BVP and ACC, interpolate ACC to BVP rate, compute activity context."""
    normalized_dir.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(subjects_dir / PARAMS_FILE).set_index('channel')
    bvp_min, bvp_max = float(params.loc['BVP', 'min']), float(params.loc['BVP', 'max'])
    acc_min, acc_max = float(params.loc['ACC_MAG', 'min']), float(params.loc['ACC_MAG', 'max'])
    bvp_range = bvp_max - bvp_min
    acc_range = acc_max - acc_min

    # Rolling window in samples on interpolated (64 Hz) ACC
    context_window = CONTEXT_WINDOW_S * BVP_RATE

    # First pass: compute raw context arrays to derive global normalization stats
    subject_cache: dict[str, dict] = {}
    all_ctx_mean: list[np.ndarray] = []
    all_ctx_std: list[np.ndarray] = []

    for subject_dir in sorted(subjects_dir.glob('S*')):
        sid = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')
        acc = np.load(subject_dir / 'acc_mag.npy')
        static = np.load(subject_dir / 'static.npy')

        norm_bvp = ((bvp - bvp_min) / bvp_range).astype(np.float32)
        norm_acc = ((acc - acc_min) / acc_range).astype(np.float32)

        acc_interp = np.interp(
            np.linspace(0, 1, len(norm_bvp)),
            np.linspace(0, 1, len(norm_acc)),
            norm_acc,
        ).astype(np.float32)

        rolling = pd.Series(acc_interp).rolling(context_window)
        ctx_mean = rolling.mean().values[context_window - 1:].astype(np.float32)
        ctx_std  = rolling.std().values[context_window - 1:].astype(np.float32)

        # Trim signals to match context length
        trimmed_bvp = norm_bvp[context_window - 1:]
        trimmed_acc = acc_interp[context_window - 1:]

        subject_cache[sid] = {
            'bvp': trimmed_bvp,
            'acc': trimmed_acc,
            'ctx_mean': ctx_mean,
            'ctx_std': ctx_std,
            'static': static,
        }
        all_ctx_mean.append(ctx_mean)
        all_ctx_std.append(ctx_std)

    # Compute global context normalization bounds
    ctx_mean_all = np.concatenate(all_ctx_mean)
    ctx_std_all  = np.concatenate(all_ctx_std)
    cm_min, cm_max = float(ctx_mean_all.min()), float(ctx_mean_all.max())
    cs_min, cs_max = float(ctx_std_all.min()),  float(ctx_std_all.max())
    cm_range = cm_max - cm_min
    cs_range = cs_max - cs_min

    # Second pass: normalize context and save everything
    for subject_id, data in subject_cache.items():
        norm_cm = ((data['ctx_mean'] - cm_min) / cm_range).astype(np.float32)
        norm_cs = ((data['ctx_std']  - cs_min) / cs_range).astype(np.float32)
        context = np.column_stack([norm_cm, norm_cs]).astype(np.float32)

        save_dir = normalized_dir / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy',     data['bvp'])
        np.save(save_dir / 'acc.npy',     data['acc'])
        np.save(save_dir / 'context.npy', context)
        np.save(save_dir / 'static.npy',  data['static'])
        print(f"  {subject_id}: {len(data['bvp'])} samples")


# ---------------------------------------------------------------------------
# Stage 3 — Synthetic anomalies on raw BVP
# ---------------------------------------------------------------------------

def inject_anomalies(
    bvp: np.ndarray,
    rng: np.random.Generator,
    anomaly_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject anomalies into a raw BVP signal at random intervals.

    Perturbations are scaled relative to the signal's own range / std so that
    they are equally disruptive regardless of the sensor's absolute output range.
    Returns (anomalous_bvp, labels) where labels is a per-sample bitmap.
    """
    result = bvp.copy()
    labels = np.zeros(len(bvp), dtype=np.float32)

    sig_range = float(bvp.max() - bvp.min())
    sig_std   = float(bvp.std())
    n = len(bvp)

    min_len = BVP_RATE * 8    # at least one 8-second window
    max_len = BVP_RATE * 60   # up to 60 seconds
    target  = int(n * anomaly_prob)

    attempts = 0
    while int(labels.sum()) < target and attempts < 10_000:
        attempts += 1
        length = int(rng.integers(min_len, min(max_len, n // 2) + 1))
        start  = int(rng.integers(0, max(1, n - length)))
        seg    = slice(start, start + length)

        if labels[seg].any():
            continue

        kind = int(rng.integers(0, 5))
        if kind == 0:   # transient spike
            scale = sig_range * float(rng.uniform(0.3, 0.8))
            result[seg] += scale * float(rng.choice([-1.0, 1.0]))
        elif kind == 1: # flatline / sensor dropout
            result[seg] = result[start]
        elif kind == 2: # amplitude blow-up around local mean
            mean = float(result[seg].mean())
            result[seg] = mean + (result[seg] - mean) * float(rng.uniform(2.0, 4.0))
        elif kind == 3: # low-frequency baseline wander
            t = np.linspace(0, float(rng.uniform(1.0, 3.0)) * np.pi, length)
            result[seg] += sig_range * 0.3 * np.sin(t + float(rng.uniform(0, np.pi)))
        else:           # noise burst
            result[seg] += rng.normal(0.0, sig_std * 0.5, size=length)

        labels[seg] = 1.0

    return result.astype(np.float32), labels


def create_anomalous_signals(subjects_dir: Path, anomalous_dir: Path):
    """Add synthetic anomalies to raw BVP from subject-signals.

    Only BVP is modified; ACC is not stored here (load from subject-signals directly).
    labels.npy is a per-sample bitmap: 1 = anomalous, 0 = clean.
    """
    anomalous_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(FEATURE_SEED)

    for subject_dir in sorted(subjects_dir.glob('S*')):
        subject_id = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')

        anomalous_bvp, labels = inject_anomalies(bvp, rng, ANOMALY_PROB)

        save_dir = anomalous_dir / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'bvp.npy',    anomalous_bvp)
        np.save(save_dir / 'labels.npy', labels)

        print(f"  {subject_id}: {len(bvp)} samples, {labels.mean():.1%} anomalous")


# ---------------------------------------------------------------------------
# Stage 4 — Feature dataset
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


def build_feature_dataset(anomalous_dir: Path, subjects_dir: Path, feature_dir: Path):
    """Window anomalous BVP and raw ACC into 8-second windows and extract features.

    BVP comes from anomalous-signals (raw with anomalies, un-normalized).
    ACC comes from subject-signals (raw magnitude, 32 Hz).
    Windows are labeled 1 if any BVP sample in the window is anomalous.
    Features are stored per subject under S*/; standardization (z-score) stats
    are still computed globally and saved at the top level for on-device use.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)

    per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    print("Building feature dataset from:", end="")
    for subject_dir in sorted(anomalous_dir.glob('S*')):
        subject_id = subject_dir.name
        bvp = np.load(subject_dir / 'bvp.npy')
        lbl = np.load(subject_dir / 'labels.npy')
        acc = np.load(subjects_dir / subject_id / 'acc_mag.npy')

        n_windows = min(
            (len(bvp) - BVP_WINDOW) // BVP_SHIFT + 1,
            (len(acc) - ACC_WINDOW) // ACC_SHIFT + 1,
        )

        features: list[np.ndarray] = []
        labels:   list[float]      = []
        for i in range(max(0, n_windows)):
            bvp_start = i * BVP_SHIFT
            acc_start = i * ACC_SHIFT

            bvp_win = bvp[bvp_start : bvp_start + BVP_WINDOW]
            acc_win = acc[acc_start : acc_start + ACC_WINDOW]
            lbl_win = lbl[bvp_start : bvp_start + BVP_WINDOW]

            features.append(extract_features(bvp_win, acc_win))
            labels.append(1.0 if lbl_win.any() else 0.0)

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

    np.save(feature_dir / 'feature_stats.npy', np.stack([mean, std]).astype(np.float32))
    print(f"Saved {total} windows ({anomalous} anomalous) to {feature_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'datasets_dir', nargs='?', type=Path, default=DEFAULT_DATASETS_DIR,
        help=f"Datasets directory (default: {DEFAULT_DATASETS_DIR})")
    args = parser.parse_args()

    datasets_dir: Path = args.datasets_dir
    raw_dir       = datasets_dir / RAW_SUBDIR
    subjects_dir  = datasets_dir / SUBJECTS_SUBDIR
    normalized_dir = datasets_dir / NORMALIZED_SUBDIR
    anomalous_dir = datasets_dir / ANOMALOUS_SUBDIR
    feature_dir   = datasets_dir / FEATURE_SUBDIR

    if raw_dir.is_dir():
        print(f"Raw dataset already present at {raw_dir}")
    else:
        download_dataset(datasets_dir, raw_dir)

    if (subjects_dir / PARAMS_FILE).exists():
        print(f"subject-signals already present at {subjects_dir}")
    else:
        print(f"\nStage 1: Extracting raw signals into {subjects_dir}/ ...")
        written = extract_subject_signals(raw_dir, subjects_dir)
        print(f"Processed {len(written)} subjects")

    if normalized_dir.is_dir() and any(normalized_dir.glob('S*')):
        print(f"normalized-signals already present at {normalized_dir}")
    else:
        print(f"\nStage 2: Normalizing signals into {normalized_dir}/ ...")
        normalize_signals(subjects_dir, normalized_dir)

    if anomalous_dir.is_dir() and any(anomalous_dir.glob('S*')):
        print(f"anomalous-signals already present at {anomalous_dir}")
    else:
        print(f"\nStage 3: Creating anomalous signals in {anomalous_dir}/ ...")
        create_anomalous_signals(subjects_dir, anomalous_dir)

    if (feature_dir / 'feature_stats.npy').exists():
        print(f"feature-anomaly already present at {feature_dir}")
    else:
        print(f"\nStage 4: Building feature dataset in {feature_dir}/ ...")
        build_feature_dataset(anomalous_dir, subjects_dir, feature_dir)
