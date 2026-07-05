from dataclasses import dataclass
from pathlib import Path

from common.db import ModelPurpose
from ml.models import (
    cnn_autoencoder,
    feature_mlp,
    gru_autoencoder,
    lstm_autoencoder,
)
from ml.models.common import Trainer, TrainerBuilder

@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    purpose: ModelPurpose
    app_version: str
    build_trainer: TrainerBuilder
    firmware_id: int | None = None
    version: str = "1.0"   # human-facing label; NOT the compatibility key


MODELS: dict[str, ModelSpec] = {
    "feature-mlp": ModelSpec(
        key="feature-mlp",
        name="Feature-based MLP",
        purpose=ModelPurpose.train_only,
        app_version="1.0.0",
        build_trainer=feature_mlp.get_trainer,
    ),
    "lstm-ae": ModelSpec(
        key="lstm-ae",
        name="LSTM Autoencoder",
        purpose=ModelPurpose.train_only,
        app_version="1.0.0",
        build_trainer=lstm_autoencoder.get_trainer,
    ),
    "gru-ae": ModelSpec(
        key="gru-ae",
        name="GRU Autoencoder",
        purpose=ModelPurpose.train_only,
        app_version="1.0.0",
        build_trainer=gru_autoencoder.get_trainer,
    ),
    "cnn-ae": ModelSpec(
        key="cnn-ae",
        name="CNN Autoencoder",
        purpose=ModelPurpose.train_only,
        app_version="1.0.0",
        build_trainer=cnn_autoencoder.get_trainer,
    ),
}

