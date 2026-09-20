import argparse
from pathlib import Path

import numpy as np

from common.config import CALIBRATION_DIR, DATASETS_DIR
from ml.dataset_list import BASES, DATASETS
from ml.model_list import MODELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download and prepare a dataset base's on-disk arrays.")
    parser.add_argument('dataset_base', choices=sorted(BASES), help='Dataset base to prepare')
    parser.add_argument(
        'datasets_dir', nargs='?', type=Path, default=DATASETS_DIR,
        help="Datasets directory")
    args = parser.parse_args()

    BASES[args.dataset_base].prepare(args.datasets_dir)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    keys = {spec.trainer_cls.calibration_key or spec.trainer_cls.training_key
            for spec in MODELS.values()}
    for key in sorted(keys):
        try:
            calibration = DATASETS[key].build(args.datasets_dir).calibration_data()
        except FileNotFoundError as exc:
            print(f"  - calibration for '{key}' skipped: {exc}")
            continue
        np.save(CALIBRATION_DIR / f'{key}.npy', calibration)
        print(f"  + calibration artifact for '{key}'")
