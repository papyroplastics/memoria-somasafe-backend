import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from ml.data import (
    RAW_SUBDIR, SUBJECTS_SUBDIR, ANOMALOUS_SUBDIR, FEATURE_SUBDIR,
    extract_subject_signals, create_anomalous_signals, build_feature_dataset,
)

DEFAULT_DATASETS_DIR = Path('datasets')
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
        'datasets_dir', nargs='?', type=Path, default=DEFAULT_DATASETS_DIR,
        help=f"Datasets directory (default: {DEFAULT_DATASETS_DIR})")
    args = parser.parse_args()

    datasets_dir: Path = args.datasets_dir
    raw_dir       = datasets_dir / RAW_SUBDIR
    subjects_dir  = datasets_dir / SUBJECTS_SUBDIR
    anomalous_dir = datasets_dir / ANOMALOUS_SUBDIR
    feature_dir   = datasets_dir / FEATURE_SUBDIR

    if raw_dir.is_dir():
        print(f"Raw dataset already present at {raw_dir}")
    else:
        download_dataset(datasets_dir, raw_dir)

    if subjects_dir.is_dir():
        print(f"subject-signals already present at {subjects_dir}")
    else:
        print(f"\nStage 1: Extracting raw signals into {subjects_dir}/ ...")
        written = extract_subject_signals(raw_dir, subjects_dir)
        print(f"Processed {len(written)} subjects")

    if anomalous_dir.is_dir() and any(anomalous_dir.glob('S*')):
        print(f"anomalous-signals already present at {anomalous_dir}")
    else:
        print(f"\nStage 2: Creating anomalous signals in {anomalous_dir}/ ...")
        create_anomalous_signals(subjects_dir, anomalous_dir)

    if feature_dir.is_dir():
        print(f"{FEATURE_SUBDIR} already present at {feature_dir}")
    else:
        print(f"\nStage 3: Building feature dataset in {feature_dir}/ ...")
        build_feature_dataset(anomalous_dir, subjects_dir, feature_dir)
