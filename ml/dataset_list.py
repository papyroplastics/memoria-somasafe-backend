from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml.sources.common import DataSource
from ml.sources.covertype import CovertypeSource, prepare_covertype
from ml.sources.dalia import (
    FEATURES, LOW_ACTIVITY, SIGNAL, VARIANTS, DaliaFeatureSource, DaliaSignalSource,
    prepare_ppg_dalia,
)
from ml.sources.mnist import IID, NONIID, MnistShardSource, MnistTestSource, prepare_mnist


@dataclass(frozen=True)
class DatasetBase:
    key: str
    name: str
    activities: tuple[int, ...] | None
    prepare: Callable[[Path], None]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    build: Callable[[Path], DataSource]


BASES: dict[str, DatasetBase] = {
    "ppg-dalia": DatasetBase(
        key="ppg-dalia", name="PPG-DaLiA (all activities)", activities=None,
        prepare=prepare_ppg_dalia),
    "ppg-dalia-low": DatasetBase(
        key="ppg-dalia-low", name="PPG-DaLiA (low-activity windows)", activities=LOW_ACTIVITY,
        prepare=prepare_ppg_dalia),
    "mnist": DatasetBase(
        key="mnist", name="MNIST", activities=None,
        prepare=prepare_mnist),
    "covertype": DatasetBase(
        key="covertype", name="Covertype", activities=None,
        prepare=prepare_covertype),
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

DATASETS["mnist-iid"] = DatasetSpec(
    key="mnist-iid", name="MNIST (IID shards)",
    build=lambda root: MnistShardSource(root, key="mnist-iid", partition=IID))
DATASETS["mnist-noniid"] = DatasetSpec(
    key="mnist-noniid", name="MNIST (non-IID Dirichlet shards)",
    build=lambda root: MnistShardSource(root, key="mnist-noniid", partition=NONIID))
DATASETS["mnist-test"] = DatasetSpec(
    key="mnist-test", name="MNIST (test split)",
    build=lambda root: MnistTestSource(root, key="mnist-test"))

DATASETS["covertype-wilderness"] = DatasetSpec(
    key="covertype-wilderness", name="Covertype (wilderness-area shards)",
    build=lambda root: CovertypeSource(root, key="covertype-wilderness"))
