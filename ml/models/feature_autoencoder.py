from pathlib import Path

import numpy as np
import tensorflow as tf

from ..layers import Dense, relu
from ..preprocessing import BVP_WINDOW, N_FEATURES
from ..sources.dalia import CLEAN
from .common import AutoencoderTrainer, TrainableAutoencoder


class FeatureAutoencoder(TrainableAutoencoder):
    """Dense autoencoder over the same hand-crafted feature vector ``feature-mlp``
    classifies, rather than over the waveform. Reconstruction error on the waveform only
    grows with how *hard* a window is to reconstruct, so a slowed rhythm — a smoother,
    more predictable wave than the physiological one — scores below the clean threshold;
    the feature vector replaces that with quantities whose normal range is bounded on
    both sides. Four dense layers, no labels."""

    def __init__(self, name: str, batch_size: int, n_features: int = N_FEATURES,
                 hidden_dim: int = 32, latent_dim: int = 6,
                 learning_rate: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, input_shape=(n_features,))
        self.n_features = n_features

        self.enc_in = Dense(n_features, hidden_dim, activation=relu)
        self.to_latent = Dense(hidden_dim, latent_dim, activation=relu)
        self.from_latent = Dense(latent_dim, hidden_dim, activation=relu)
        self.dec_out = Dense(hidden_dim, n_features)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, features):
        return self.dec_out(self.from_latent(self.to_latent(self.enc_in(features))))

    def eval_eager(self, features: tf.Tensor):
        return self._eval_core(features)

    def train_eager(self, features: tf.Tensor):
        return self._train_core(features)


class FeatureAutoencoderTrainer(AutoencoderTrainer):
    """Feeds feature vectors where AutoencoderTrainer feeds windows, on the
    non-overlapping grid the labels and the feature vectors live on."""

    dataset_tensors = ['features']

    def __init__(self, model: FeatureAutoencoder, data_root: Path):
        super().__init__(model, data_root, shift=BVP_WINDOW)
        self.model: FeatureAutoencoder = model  # type: ignore

    def subject_arrays(self, sid):
        return (self.data.features(sid, CLEAN),)

    def calibration_arrays(self):
        return self.calibration.calibration_features()

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        for batch in eval_dataset.take(1):
            target = np.asarray(batch[0])[0]
            recon = np.asarray(self.model.eval(*batch)['reconstruction'])[0]
            index = np.arange(len(target))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(index - 0.2, target, 0.4, label='features')
            ax.bar(index + 0.2, recon, 0.4, label='reconstruction')
            ax.set_xticks(index)
            ax.set_xlabel('feature index')
            ax.set_ylabel('z-score (per subject)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(result_dir / 'reconstruction.png')
            plt.close(fig)
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break


def get_model(data_root: Path, batch_size: int | None = None) -> FeatureAutoencoder:
    return FeatureAutoencoder(
        name='dalia_feature_ae',
        batch_size=batch_size or FeatureAutoencoder.default_batch_size,
    )


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    return FeatureAutoencoderTrainer(get_model(data_root, batch_size), data_root)
