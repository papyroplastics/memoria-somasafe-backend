from dataclasses import dataclass

from common.db import SubmissionType
from ml.models import (
    cnn_autoencoder,
    feature_autoencoder,
    feature_mlp,
    gru_autoencoder,
    lstm_autoencoder,
    mnist_mlp,
)
from ml.models.common import Trainer, TrainableModel
from ml.models.signal import AutoencoderTrainer


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    min_app_version: str
    model_cls: type[TrainableModel]
    trainer_cls: type[Trainer]
    submission_type: SubmissionType
    firmware_id: int | None = None
    version: int = 1
    contract_version: int = 0
    artifacts_key: str | None = None

    @property
    def artifact_key(self) -> str:
        return self.artifacts_key or self.key


MODELS: dict[str, ModelSpec] = {
    "feature-mlp": ModelSpec(
        key="feature-mlp",
        name="Feature-based MLP",
        min_app_version="1.0.0",
        model_cls=feature_mlp.FeatureMLP,
        trainer_cls=feature_mlp.FeatureMLPTrainer,
        submission_type=SubmissionType.quantize,
        contract_version=1,
    ),
    "lstm-ae": ModelSpec(
        key="lstm-ae",
        name="LSTM Autoencoder",
        min_app_version="1.0.0",
        model_cls=lstm_autoencoder.LSTMAutoencoder,
        trainer_cls=AutoencoderTrainer,
        submission_type=SubmissionType.raw,
    ),
    "gru-ae": ModelSpec(
        key="gru-ae",
        name="GRU Autoencoder",
        min_app_version="1.0.0",
        model_cls=gru_autoencoder.GRUAutoencoder,
        trainer_cls=AutoencoderTrainer,
        submission_type=SubmissionType.raw,
    ),
    "feature-ae": ModelSpec(
        key="feature-ae",
        name="Feature Autoencoder",
        min_app_version="1.0.0",
        model_cls=feature_autoencoder.FeatureAutoencoder,
        trainer_cls=feature_autoencoder.FeatureAutoencoderTrainer,
        submission_type=SubmissionType.raw,
    ),
    "feature-ae-secure": ModelSpec(
        key="feature-ae-secure",
        name="Feature Autoencoder (secure)",
        min_app_version="1.0.0",
        model_cls=feature_autoencoder.FeatureAutoencoder,
        trainer_cls=feature_autoencoder.FeatureAutoencoderTrainer,
        submission_type=SubmissionType.secure,
        artifacts_key="feature-ae",
    ),
    "cnn-ae": ModelSpec(
        key="cnn-ae",
        name="CNN Autoencoder",
        min_app_version="1.0.0",
        model_cls=cnn_autoencoder.CNNAutoencoder,
        trainer_cls=AutoencoderTrainer,
        submission_type=SubmissionType.raw,
    ),
    "mnist-mlp": ModelSpec(
        key="mnist-mlp",
        name="MNIST MLP",
        min_app_version="1.0.0",
        model_cls=mnist_mlp.MnistMLP,
        trainer_cls=mnist_mlp.MnistMLPTrainer,
        submission_type=SubmissionType.quantize,
        contract_version=1,
    ),
}
