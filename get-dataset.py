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
PROCESSED_SUBDIR = 'ppg-dalia-processed'
SUBJECTS_SUBDIR = 'subjects'
FEATURE_SUBDIR = 'feature-anomaly'
PARAMS_FILE = 'params.csv'

DATASET_URL = 'https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip'

CHANNELS = ['BVP', 'ACC']

SAMPLE_RATE = 64                  # hz
WINDOW_SIZE = SAMPLE_RATE * 8     # 8 s windows
SHIFT = SAMPLE_RATE * 3           # 3 s stride
ANOMALY_PROB = 0.5
FEATURE_SEED = 1234
N_FEATURES = 17  # keep in sync with extract_features


def download_dataset(datasets_dir: Path, raw_dir: Path):
    datasets_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
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


def load_raw_data_pickle(raw_dir: Path, subject_id: int) -> dict | None:
    path = raw_dir / f'S{subject_id}' / f'S{subject_id}.pkl'
    if not path.exists():
        return None

    with open(path, 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    return data


def user_description_vector(quest: dict) -> np.ndarray:
    gender_raw = quest.get('Gender', 'm').strip().lower()
    gender_norm = 1.0 if gender_raw == 'f' else 0.0

    age_norm    = (quest.get('AGE',     30) -  20) / 20   # 20  - 40
    height_norm = (quest.get('HEIGHT', 150) - 100) / 100  # 100 - 200
    weight_norm = (quest.get('WEIGHT',  70) -  40) / 110  # 40  - 150
    skin_norm   = (quest.get('SKIN',     3) -   1) / 5    # 1   - 6
    sport_norm  = (quest.get('SPORT',    3) -   1) / 6    # 1   - 6

    return np.array([
        gender_norm,
        age_norm,
        height_norm,
        weight_norm,
        skin_norm,
        sport_norm,
    ], dtype=np.float32)


def extract_stage_1(
    raw_dir: Path, processed_dir: Path, subject_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    print(f"--- Processing Subject S{subject_id} (stage 1) ---")

    raw_data = load_raw_data_pickle(raw_dir, subject_id)
    if raw_data is None:
        print(f"   File not found for subject {subject_id}")
        return None

    wrist = raw_data['signal']['wrist']
    bvp = wrist['BVP'].flatten()
    target_len = len(bvp)

    acc_g = wrist['ACC'] / 64.0
    acc_mag = np.sqrt(np.sum(acc_g**2, axis=1))
    acc_mag = np.convolve(acc_mag, np.ones(5), mode='same')
    acc = np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(acc_mag)), acc_mag)

    signal_matrix = np.stack([bvp, acc], axis=1)

    # Activity context represents the movement in the last two minutes
    window_samples = 2 * 60 * 64
    rolling = pd.Series(acc).rolling(window_samples)

    act_mean: np.ndarray = rolling.mean().values
    act_std: np.ndarray = rolling.std().values
    activity_context_matrix = np.column_stack((act_mean, act_std))

    # First window_samples values will be null after rooling ops so we cull them
    activity_context_matrix = activity_context_matrix [window_samples-1:]
    signal_matrix = signal_matrix [window_samples-1:]

    static_vector = user_description_vector(raw_data['questionnaire'])

    save_dir = processed_dir / f"S{subject_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / 'context.npy', activity_context_matrix.astype(np.float32))
    np.save(save_dir / 'static.npy', static_vector.astype(np.float32))

    bounds = np.stack([signal_matrix.min(axis=0), signal_matrix.max(axis=0)], axis=1)

    print(f"   Shape: {signal_matrix.shape}")
    return signal_matrix, bounds


def extract_stage_2(
    processed_dir: Path, subject_id: int,
    signal_matrix: np.ndarray, global_min: np.ndarray, global_max: np.ndarray,
):
    print(f"--- Processing Subject S{subject_id} (stage 2) ---")

    normalized_matrix = (signal_matrix - global_min) / (global_max - global_min)

    save_dir = processed_dir / f"S{subject_id}"
    np.save(save_dir / 'signal.npy', normalized_matrix.astype(np.float32))
    print(f"   Saved to {save_dir}")


def extract_signals(raw_dir: Path, processed_dir: Path) -> list[int]:
    stage1: dict[int, np.ndarray] = {}
    global_min: np.ndarray | None = None
    global_max: np.ndarray | None = None

    subject_ids = sorted(
        int(p.name[1:]) for p in raw_dir.glob('S*')
        if p.is_dir() and p.name[1:].isdigit()
    )

    for subject_id in subject_ids:
        try:
            result = extract_stage_1(raw_dir, processed_dir, subject_id)
        except Exception as e:
            print(f"Error processing S{subject_id}: {e}")
            continue
        if result is None:
            continue

        signal_matrix, bounds = result
        stage1[subject_id] = signal_matrix
        subject_min, subject_max = bounds[:, 0], bounds[:, 1]
        global_min = subject_min if global_min is None else np.minimum(global_min, subject_min)
        global_max = subject_max if global_max is None else np.maximum(global_max, subject_max)

    if global_min is None or global_max is None:
        return []

    pd.DataFrame({
        'channel': CHANNELS,
        'min': global_min,
        'max': global_max,
    }).to_csv(processed_dir / PARAMS_FILE, index=False)

    for subject_id, signal_matrix in stage1.items():
        extract_stage_2(processed_dir, subject_id, signal_matrix, global_min, global_max)

    return list(stage1)


# --- Synthetic anomalies + feature extraction for the FeatureMLP classifier ---

def inject_anomaly(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Corrupt the BVP channel of a clean window to simulate an anomaly. The
    signal is min-max normalized to ~[0, 1], so the perturbations below are
    large relative to the clean dynamic range and clearly separable."""
    w = window.copy()
    length = w.shape[0]
    seg_len = int(rng.integers(length // 8, length // 2))
    start = int(rng.integers(0, length - seg_len))
    seg = slice(start, start + seg_len)
    kind = int(rng.integers(0, 5))

    if kind == 0:        # transient spike
        w[seg, 0] += rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
    elif kind == 1:      # flatline / sensor dropout
        w[seg, 0] = w[start, 0]
    elif kind == 2:      # amplitude blow-up around the segment mean
        mean = w[seg, 0].mean()
        w[seg, 0] = mean + (w[seg, 0] - mean) * rng.uniform(2.0, 4.0)
    elif kind == 3:      # low-frequency baseline wander over the whole window
        t = np.linspace(0, rng.uniform(1.0, 3.0) * np.pi, length)
        w[:, 0] += 0.3 * np.sin(t + rng.uniform(0, np.pi))
    else:                # localized noise burst
        w[seg, 0] += rng.normal(0.0, 0.25, size=seg_len)

    return w


def extract_features(window: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cheap, on-device-replicable feature vector for a [L, 2] (BVP, ACC) window."""
    feats: list[float] = []
    for channel in (window[:, 0], window[:, 1]):
        feats += [
            float(channel.mean()),
            float(channel.std()),
            float(channel.min()),
            float(channel.max()),
            float(channel.max() - channel.min()),
            float(np.sqrt(np.mean(channel ** 2))),
            float(np.mean(np.abs(np.diff(channel)))),
        ]

    bvp = window[:, 0] - window[:, 0].mean()
    feats.append(float(np.mean(np.abs(np.diff(np.sign(bvp)))) / 2))  # zero-crossing rate

    spectrum = np.abs(np.fft.rfft(bvp))
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / sample_rate)
    feats.append(float(freqs[np.argmax(spectrum)]))                  # dominant frequency
    band = (freqs >= 0.7) & (freqs <= 3.5)                           # plausible HR band
    feats.append(float(spectrum[band].sum() / (spectrum.sum() + 1e-8)))

    return np.asarray(feats, dtype=np.float32)


def build_feature_cache(subjects_dir: Path, feature_dir: Path, window_size: int,
                        shift: int, sample_rate: int, anomaly_prob: float, seed: int):
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[float] = []

    print("Building feature/anomaly dataset from:", end="")
    for subject_dir in sorted(subjects_dir.glob('S*')):
        signal = np.load(subject_dir / 'signal.npy')
        window_count = (len(signal) - window_size) // shift + 1
        for i in range(window_count):
            window = signal[i * shift: i * shift + window_size]
            if rng.random() < anomaly_prob:
                window = inject_anomaly(window, rng)
                labels.append(1.0)
            else:
                labels.append(0.0)
            features.append(extract_features(window, sample_rate))
        print(f" {subject_dir.name}", end="", flush=True)
    print()

    x = np.stack(features)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)

    # Standardize features and persist the stats so on-device extraction can
    # reproduce the exact same normalization.
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    x = ((x - mean) / std).astype(np.float32)

    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / 'features.npy', x)
    np.save(feature_dir / 'labels.npy', y)
    np.save(feature_dir / 'feature_stats.npy', np.stack([mean, std]).astype(np.float32))
    print(f"Saved {len(y)} windows ({int(y.sum())} anomalous) to {feature_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'datasets_dir', nargs='?', type=Path, default=DEFAULT_DATASETS_DIR,
        help=f"Datasets directory (default: {DEFAULT_DATASETS_DIR})")
    args = parser.parse_args()

    datasets_dir: Path = args.datasets_dir
    raw_dir = datasets_dir / RAW_SUBDIR
    processed_dir = datasets_dir / PROCESSED_SUBDIR
    subjects_dir = processed_dir / SUBJECTS_SUBDIR
    feature_dir = processed_dir / FEATURE_SUBDIR

    if raw_dir.is_dir():
        print(f"Raw dataset already present at {raw_dir}")
    else:
        download_dataset(datasets_dir, raw_dir)

    if subjects_dir.is_dir():
        print(f"Processed subjects already present at {subjects_dir}")
    else:
        subjects_dir.mkdir(parents=True, exist_ok=True)
        written = extract_signals(raw_dir, subjects_dir)
        print(f"\nProcessed {len(written)} subjects into {subjects_dir}")

    if (feature_dir / 'features.npy').exists():
        print(f"Feature/anomaly dataset already present at {feature_dir}")
    else:
        build_feature_cache(
            subjects_dir, feature_dir, window_size=WINDOW_SIZE, shift=SHIFT,
            sample_rate=SAMPLE_RATE, anomaly_prob=ANOMALY_PROB, seed=FEATURE_SEED,
        )

