"""Single source of truth for the models the system knows about.

Each entry pairs the model's static metadata (served by the gateway via the DB)
with the TensorFlow trainer builder used to materialize it. The architecture
fingerprint is *not* stored here — it is derived from a built model on demand
(``trainer.model.arch_fingerprint()``) so it can never drift from the code:

  - scripts.train builds a single trainer and ignores fingerprints.
  - scripts.db_seed builds every (trained) model to upload its fingerprint.
  - worker.tasks builds every available model once at startup.

This module imports TensorFlow (via the model packages); the api package must
not import it — the gateway trusts what scripts.db_seed wrote to the database.
"""

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
        key="feature-mlp", name="Feature-based MLP",
        purpose=ModelPurpose.train_only, app_version="1.0.0",
        build_trainer=feature_mlp.get_trainer,
    ),
    "lstm-ae": ModelSpec(
        key="lstm-ae", name="LSTM Autoencoder",
        purpose=ModelPurpose.train_only, app_version="1.0.0",
        build_trainer=lstm_autoencoder.get_trainer,
    ),
    "gru-ae": ModelSpec(
        key="gru-ae", name="GRU Autoencoder",
        purpose=ModelPurpose.train_only, app_version="1.0.0",
        build_trainer=gru_autoencoder.get_trainer,
    ),
    "cnn-ae": ModelSpec(
        key="cnn-ae", name="CNN Autoencoder",
        purpose=ModelPurpose.train_only, app_version="1.0.0",
        build_trainer=cnn_autoencoder.get_trainer,
    ),
}

def build_fingerprinted(key: str) -> tuple[Trainer, str]:
    """Build a trainer and compute its architecture fingerprint from the model."""
    trainer = MODELS[key].build_trainer()
    return trainer, trainer.model.arch_fingerprint()
