from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml.sources.common import DataSource
from ml.sources.dalia import (
    CLEAN, FEATURES, LOW_ACTIVITY, SIGNAL, VARIANTS, DaliaFeatureSource, DaliaSignalSource,
)


@dataclass(frozen=True)
class DatasetBase:
    key: str
    name: str
    activities: tuple[int, ...] | None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    build: Callable[[Path], DataSource]


BASES: dict[str, DatasetBase] = {
    "ppg-dalia": DatasetBase(
        key="ppg-dalia", name="PPG-DaLiA (all activities)", activities=None),
    "ppg-dalia-low": DatasetBase(
        key="ppg-dalia-low", name="PPG-DaLiA (low-activity windows)", activities=LOW_ACTIVITY),
}

DATASETS: dict[str, DatasetSpec] = {}
for _base in BASES.values():
    for _variant in VARIANTS:
        for _modality, _cls in ((SIGNAL, DaliaSignalSource), (FEATURES, DaliaFeatureSource)):
            _key = f"{_base.key}-{_modality}-{_variant}"
            DATASETS[_key] = DatasetSpec(
                key=_key,
                name=f"{_base.name} — {_modality} ({_variant})",
                build=lambda root, cls=_cls, variant=_variant, activities=_base.activities, key=_key:
                    cls(root, key=key, variant=variant, activities=activities),
            )

TRAINING_DATASET = 'ppg-dalia-low'
CALIBRATION_DATASET = 'ppg-dalia'

def training_source(data_root: Path, variant_suffix: str) -> DataSource:
    """``variant_suffix`` is e.g. ``"signal-clean"`` or ``"features-mixed"``."""
    return DATASETS[f'{TRAINING_DATASET}-{variant_suffix}'].build(data_root)


def calibration_source(data_root: Path, variant_suffix: str) -> DataSource:
    """Always the clean variant of the unfiltered dataset, same modality as training —
    the device runs everywhere, so the int8 tensor scales must cover the full range."""
    modality = variant_suffix.split('-', 1)[0]
    return DATASETS[f'{CALIBRATION_DATASET}-{modality}-{CLEAN}'].build(data_root)
