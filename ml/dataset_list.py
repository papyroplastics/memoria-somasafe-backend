from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml.preprocessing import LOW_ACTIVITY
from ml.sources.common import DataSource
from ml.sources.dalia import DaliaSource


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    # Every dataset brings its own preprocessing and its own on-disk layout; this is the
    # only thing consumers need to read one.
    build: Callable[[Path], DataSource]


DATASETS: dict[str, DatasetSpec] = {
    "ppg-dalia": DatasetSpec(
        key="ppg-dalia",
        name="PPG-DaLiA (all activities)",
        build=lambda root: DaliaSource(root, key="ppg-dalia"),
    ),
    "ppg-dalia-low": DatasetSpec(
        key="ppg-dalia-low",
        name="PPG-DaLiA (low-activity windows)",
        build=lambda root: DaliaSource(root, key="ppg-dalia-low",
                                       activities=LOW_ACTIVITY),
    ),
}

# What every Trainer trains and evaluates on. Consumers of Trainer are the system itself
# (train.py, fed_client.py, worker.tasks), which has no reason to pick a dataset; the
# scripts that do care build a source from DATASETS themselves.
TRAINING_DATASET = 'ppg-dalia-low'

# int8 calibration always sees every activity: the device runs everywhere, so the tensor
# scales must cover the full range, not only the range the model was trained on.
CALIBRATION_DATASET = 'ppg-dalia'


def training_source(data_root: Path) -> DataSource:
    return DATASETS[TRAINING_DATASET].build(data_root)


def calibration_source(data_root: Path) -> DataSource:
    return DATASETS[CALIBRATION_DATASET].build(data_root)
