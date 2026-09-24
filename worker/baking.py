from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sqlmodel import Session

from common.compression import compress
from common.config import SERVER_PRIVATE_KEY_FILE
from common.db import Artifact, GlobalWeights, WeightsArtifact
from ml.payload import sign_model
from ml.saving import get_optimized_model, get_trainable_model
from worker.phases import Phases
from worker.runtime import Runtime


@dataclass(frozen=True)
class Baked:
    weights: bytes
    trainable: bytes
    trainable_signature: bytes
    quantized: bytes
    quantized_signature: bytes


def bake_quantized(runtime: Runtime, weights: np.ndarray, contract_version: int,
                   phases: Phases) -> tuple[bytes, bytes]:
    with phases("restore"):
        runtime.model.restore(tf.constant(weights, dtype=tf.float32))
    with phases("get_optimized_model"):
        quantized = bytes(get_optimized_model(runtime.model, runtime.rep_dataset))
    with phases("sign_model"):
        signature = sign_model(quantized, contract_version, SERVER_PRIVATE_KEY_FILE)
    with phases("compress"):
        return compress(quantized), signature


def bake(runtime: Runtime, weights: np.ndarray, contract_version: int,
         phases: Phases) -> Baked:
    quantized, quantized_signature = bake_quantized(runtime, weights, contract_version, phases)
    with phases("get_trainable_model"):
        trainable = bytes(get_trainable_model(runtime.model))
    with phases("sign_model"):
        trainable_signature = sign_model(trainable, contract_version, SERVER_PRIVATE_KEY_FILE)
    with phases("compress"):
        return Baked(compress(weights.astype(np.float32).tobytes()), compress(trainable),
                     trainable_signature, quantized, quantized_signature)


def store(session: Session, model_key: str, version_id: int, parent_weights_id: int,
          baked: Baked) -> GlobalWeights:
    snapshot = GlobalWeights(model_key=model_key, version_id=version_id,
                             parent_weights_id=parent_weights_id, weights=baked.weights)
    session.add(snapshot)
    session.flush()
    session.add(WeightsArtifact(weights_id=snapshot.id, artifact=Artifact.trainable,
                                data=baked.trainable, signature=baked.trainable_signature))
    session.add(WeightsArtifact(weights_id=snapshot.id, artifact=Artifact.quantized,
                                data=baked.quantized, signature=baked.quantized_signature))
    return snapshot
