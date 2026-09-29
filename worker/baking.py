from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
import tensorflow as tf
from sqlmodel import Session

from common.compression import compress
from common.config import SERVER_PRIVATE_KEY_FILE
from common.db import Artifact, GlobalWeights, WeightsArtifact
from ml.models.common import TrainableModel
from ml.payload import sign_model
from ml.saving import get_optimized_model, get_trainable_model
from worker.metrics import Timer


class Restore(StrEnum):
    restore = auto()


class QuantizedBake(StrEnum):
    optimize = "quantized_optimize"
    sign = "quantized_sign"
    compress = "quantized_compress"


class TrainableBake(StrEnum):
    export = "trainable_export"
    sign = "trainable_sign"
    compress = "trainable_compress"


@dataclass(frozen=True)
class Signed:
    data: bytes
    signature: bytes


def restore(model: TrainableModel, weights: np.ndarray, timer: Timer[Restore]) -> None:
    with timer(Restore.restore):
        model.restore(tf.constant(weights, dtype=tf.float32))


def bake_quantized(model: TrainableModel, rep_dataset: tf.data.Dataset, contract_version: int,
                   timer: Timer[QuantizedBake]) -> Signed:
    with timer(QuantizedBake.optimize):
        quantized = bytes(get_optimized_model(model, rep_dataset))
    with timer(QuantizedBake.sign):
        signature = sign_model(quantized, contract_version, SERVER_PRIVATE_KEY_FILE)
    with timer(QuantizedBake.compress):
        return Signed(compress(quantized), signature)


def bake_trainable(model: TrainableModel, contract_version: int,
                   timer: Timer[TrainableBake]) -> Signed:
    with timer(TrainableBake.export):
        trainable = bytes(get_trainable_model(model))
    with timer(TrainableBake.sign):
        signature = sign_model(trainable, contract_version, SERVER_PRIVATE_KEY_FILE)
    with timer(TrainableBake.compress):
        return Signed(compress(trainable), signature)


def compress_weights(weights: np.ndarray) -> bytes:
    return compress(weights.astype(np.float32).tobytes())


def store(session: Session, model_key: str, version_id: int, parent_weights_id: int,
          weights: bytes, trainable: Signed, quantized: Signed) -> GlobalWeights:
    snapshot = GlobalWeights(model_key=model_key, version_id=version_id,
                             parent_weights_id=parent_weights_id, weights=weights)
    session.add(snapshot)
    session.flush()
    session.add(WeightsArtifact(weights_id=snapshot.id, artifact=Artifact.trainable,
                                data=trainable.data, signature=trainable.signature))
    session.add(WeightsArtifact(weights_id=snapshot.id, artifact=Artifact.quantized,
                                data=quantized.data, signature=quantized.signature))
    return snapshot
