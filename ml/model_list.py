from dataclasses import dataclass

from common.db import SubmissionType
from ml.models import (
    cnn_autoencoder,
    feature_mlp,
    gru_autoencoder,
    lstm_autoencoder,
    spectral_autoencoder,
)
from ml.models.common import ModelBuilder, TrainerBuilder

@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    min_app_version: str   # oldest app that can use the current version
    build_trainer: TrainerBuilder
    # The bare model, for consumers that already know the architecture and load their own
    # data (the figure scripts) — a Trainer would only pin them to the training dataset.
    build_model: ModelBuilder
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
        build_model=feature_mlp.get_model,
        submission_type=SubmissionType.quantize,
    ),
    "lstm-ae": ModelSpec(
        key="lstm-ae",
        name="LSTM Autoencoder",
        min_app_version="1.0.0",
        build_trainer=lstm_autoencoder.get_trainer,
        build_model=lstm_autoencoder.get_model,
        submission_type=SubmissionType.raw,
    ),
    "gru-ae": ModelSpec(
        key="gru-ae",
        name="GRU Autoencoder",
        min_app_version="1.0.0",
        build_trainer=gru_autoencoder.get_trainer,
        build_model=gru_autoencoder.get_model,
        submission_type=SubmissionType.raw,
    ),
    "spectral-ae": ModelSpec(
        key="spectral-ae",
        name="Spectral Descriptor Autoencoder",
        min_app_version="1.0.0",
        build_trainer=spectral_autoencoder.get_trainer,
        build_model=spectral_autoencoder.get_model,
        submission_type=SubmissionType.raw,
    ),
    "cnn-ae": ModelSpec(
        key="cnn-ae",
        name="CNN Autoencoder",
        min_app_version="1.0.0",
        build_trainer=cnn_autoencoder.get_trainer,
        build_model=cnn_autoencoder.get_model,
        submission_type=SubmissionType.secure,
    ),
}

