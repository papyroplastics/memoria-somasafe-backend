"""
Download + sequence the (idempotent) preprocessing stages in ml/data.py:

Stage 1 → datasets/clean-signals/S*/    raw BVP (64 Hz) + ACC mag (32 Hz),
z-scored demographics (static.npy); global BVP/ACC mean/std
(norm-params.npy) + demographics mean/std (static_norm_params.npy)
Stage 2 → datasets/anomalous-signals/<kind>/S*/  per-type fully-anomalous BVP (one kind
applied to every window) — for isolated per-kind testing in autoencoder_test.py
Stage 3 → datasets/mixed-signals/S*/      raw BVP with a window-aligned ~50% mix of anomaly
kinds + per-window binary label bitmap (labels.npy) — the realistic training/distill set
Stage 4 → datasets/mixed-features/S*/     per-subject non-overlapping 8 s windowed feature
vectors + labels (from mixed-signals), global standardization stats at the top level
Signals are stored raw; z-score normalization happens at load time (no
normalized copy on disk), so signal windows align 1:1 with the feature windows.
"""

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from common.config import DATASETS_DIR

from ml.data import (
    RAW_SUBDIR, CLEAN_SUBDIR, ANOMALOUS_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR,
    extract_subject_signals, create_anomalous_signals, create_mixed_signals,
    build_feature_dataset, get_sorted_paths
)

DATASET_URL = 'https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip'

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'datasets_dir', nargs='?', type=Path, default=DATASETS_DIR,
        help=f"Datasets directory (default: {DATASETS_DIR})")
    args = parser.parse_args()

    datasets_dir: Path = args.datasets_dir
    raw_dir       = datasets_dir / RAW_SUBDIR
    subjects_dir  = datasets_dir / CLEAN_SUBDIR
    anomalous_dir = datasets_dir / ANOMALOUS_SUBDIR
    mixed_dir     = datasets_dir / MIXED_SUBDIR
    feature_dir   = datasets_dir / MIXED_FEATURE_SUBDIR

    if raw_dir.is_dir():
        print(f"Raw dataset already present at {raw_dir}")
    else:
        download_dataset(datasets_dir, raw_dir)

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

    if mixed_dir.is_dir() and any(mixed_dir.glob('S*')):
        print(f"{MIXED_SUBDIR} already present at {mixed_dir}")
    else:
        print(f"\nStage 3: Creating mixed-anomaly signals in {mixed_dir}/ ...")
        create_mixed_signals(subjects_dir, mixed_dir)

    if feature_dir.is_dir():
        print(f"{MIXED_FEATURE_SUBDIR} already present at {feature_dir}")
    else:
        print(f"\nStage 4: Building feature dataset in {feature_dir}/ ...")
        build_feature_dataset(mixed_dir, subjects_dir, feature_dir)
