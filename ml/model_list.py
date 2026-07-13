from dataclasses import dataclass
from pathlib import Path

from common.db import SubmissionType
from ml.models import (
    cnn_autoencoder,
    feature_mlp,
    gru_autoencoder,
    lstm_autoencoder,
)
from ml.models.common import TrainerBuilder

@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    min_app_version: str   # oldest app that can use the current version
    build_trainer: TrainerBuilder
    # Upload path + aggregation strategy for this model's weight updates.
    submission_type: SubmissionType
    firmware_id: int | None = None
    # Hand-bumped on any change that invalidates existing weights (architecture,
    # baked norm params, contract). The seed script errors if the fingerprint
    # moved without a bump, and publishes a new frozen-history ModelVersion row
    # when it did.
    version: int = 1


MODELS: dict[str, ModelSpec] = {
    "feature-mlp": ModelSpec(
        key="feature-mlp",
        name="Feature-based MLP",
        min_app_version="1.0.0",
        build_trainer=feature_mlp.get_trainer,
        submission_type=SubmissionType.quantize,
    ),
    "lstm-ae": ModelSpec(
        key="lstm-ae",
        name="LSTM Autoencoder",
        min_app_version="1.0.0",
        build_trainer=lstm_autoencoder.get_trainer,
        submission_type=SubmissionType.raw,
    ),
    "gru-ae": ModelSpec(
        key="gru-ae",
        name="GRU Autoencoder",
        min_app_version="1.0.0",
        build_trainer=gru_autoencoder.get_trainer,
        submission_type=SubmissionType.raw,
    ),
    "cnn-ae": ModelSpec(
        key="cnn-ae",
        name="CNN Autoencoder",
        min_app_version="1.0.0",
        build_trainer=cnn_autoencoder.get_trainer,
        submission_type=SubmissionType.secure,
    ),
}

