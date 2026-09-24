from dataclasses import dataclass

import tensorflow as tf

from common.config import DATASETS_DIR
from ml.model_list import MODELS
from ml.models.common import TrainableModel, arch_fingerprint


@dataclass(frozen=True)
class Runtime:
    model: TrainableModel
    rep_dataset: tf.data.Dataset
    fingerprint: str


_cache: dict[tuple[type, type], Runtime] = {}


def get(key: str) -> Runtime:
    spec = MODELS.get(key)
    if spec is None:
        raise ValueError(f"model '{key}' is not registered")
    cache_key = (spec.model_cls, spec.trainer_cls)
    if cache_key not in _cache:
        trainer = spec.trainer_cls(spec.model_cls(), DATASETS_DIR)
        _cache[cache_key] = Runtime(trainer.model, trainer.representative_dataset(),
                                    arch_fingerprint(trainer.model))
    return _cache[cache_key]
