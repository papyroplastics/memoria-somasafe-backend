from dataclasses import dataclass

from common.db import SubmissionType
from ml.models import (
    cnn_autoencoder,
    feature_autoencoder,
    feature_mlp,
    gru_autoencoder,
    lstm_autoencoder,
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
    version: int = 1
    contract_version: int = 0
    # Set when this entry is a second registry key over an architecture another entry
    # already owns, so it is served and seeded from that key's exported artifacts instead
    # of its own. The two keys differ only in the upload path their weight updates take.
    artifacts_key: str | None = None

    @property
    def artifact_key(self) -> str:
        return self.artifacts_key or self.key


MODELS: dict[str, ModelSpec] = {
    "feature-mlp": ModelSpec(
        key="feature-mlp",
        name="Feature-based MLP",
        min_app_version="1.0.0",
        build_trainer=feature_mlp.get_trainer,
        build_model=feature_mlp.get_model,
        submission_type=SubmissionType.quantize,
        contract_version=1,
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
    "feature-ae": ModelSpec(
        key="feature-ae",
        name="Feature Autoencoder",
        min_app_version="1.0.0",
        build_trainer=feature_autoencoder.get_trainer,
        build_model=feature_autoencoder.get_model,
        submission_type=SubmissionType.raw,
    ),
    # Same architecture and same seeded weights as 'feature-ae', reached through the
    # secure upload path, so the two integration runs differ only in that path.
    "feature-ae-secure": ModelSpec(
        key="feature-ae-secure",
        name="Feature Autoencoder (secure)",
        min_app_version="1.0.0",
        build_trainer=feature_autoencoder.get_trainer,
        build_model=feature_autoencoder.get_model,
        submission_type=SubmissionType.secure,
        artifacts_key="feature-ae",
    ),
    "cnn-ae": ModelSpec(
        key="cnn-ae",
        name="CNN Autoencoder",
        min_app_version="1.0.0",
        build_trainer=cnn_autoencoder.get_trainer,
        build_model=cnn_autoencoder.get_model,
        submission_type=SubmissionType.raw,
    ),
}

