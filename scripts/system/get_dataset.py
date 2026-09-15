import argparse
from pathlib import Path

from common.config import DATASETS_DIR
from ml.dataset_list import BASES

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download and prepare a dataset base's on-disk arrays.")
    parser.add_argument('dataset_base', choices=sorted(BASES), help='Dataset base to prepare')
    parser.add_argument(
        'datasets_dir', nargs='?', type=Path, default=DATASETS_DIR,
        help="Datasets directory")
    args = parser.parse_args()

    BASES[args.dataset_base].prepare(args.datasets_dir)
