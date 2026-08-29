from pathlib import Path

import numpy as np
import tensorflow as tf

from ..layers import Dense, relu
from ..preprocessing import BVP_WINDOW
from ..spectral import N_DESCRIPTORS, descriptor
from .common import TrainableAutoencoder, AutoencoderTrainer, autoencoder_norm_params, descriptor_norm_params


class SpectralAutoencoder(TrainableAutoencoder):
    """Dense autoencoder over the fixed spectral descriptor of an 8-second BVP window.

    Same detector contract as any other autoencoder here — raw window in, reconstruction
    error out, trained on normal windows only — but the thing being reconstructed is the
    ``ml.spectral`` descriptor rather than the waveform. A waveform autoencoder scores a
    window by how much high-frequency detail it failed to copy, which makes its error a
    complexity measure: anomalies that smooth or slow the signal reconstruct *better*
    than normal windows and land below the threshold. The descriptor's coordinates
    (pulse rate, spectral shape, waveform regularity) instead have a bounded normal
    range, so a bottleneck that learns to reproduce it fails in both directions.

    Small by construction: the descriptor is ``N_DESCRIPTORS`` numbers, so the whole
    trainable model is four dense layers."""

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 signal_mean, signal_std, descriptor_mean, descriptor_std,
                 n_signals: int = 1, hidden_dim: int = 32, latent_dim: int = 6,
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_outputs=N_DESCRIPTORS,
                         diff_weight=0.0,
                         signal_mean=signal_mean, signal_std=signal_std)

        self.descriptor_mean = tf.constant(descriptor_mean, dtype=tf.float32)
        self.descriptor_std = tf.constant(descriptor_std, dtype=tf.float32)

        self.enc_in = Dense(N_DESCRIPTORS, hidden_dim, activation=relu)
        self.to_latent = Dense(hidden_dim, latent_dim, activation=relu)
        self.from_latent = Dense(latent_dim, hidden_dim, activation=relu)
        self.dec_out = Dense(hidden_dim, N_DESCRIPTORS)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _descriptor(self, signal: tf.Tensor) -> tf.Tensor:
        """The z-scored descriptor. Detached: the transform is fixed, so it is the
        autoencoder's input and target, never something gradients flow back through."""
        raw = descriptor(signal, self.seq_len)
        return tf.stop_gradient((raw - self.descriptor_mean) / self.descriptor_std)

    def _forward(self, features: tf.Tensor) -> tf.Tensor:
        return self.dec_out(self.from_latent(self.to_latent(self.enc_in(features))))

    def _eval_core(self, signal: tf.Tensor):
        target = self._descriptor(signal)
        reconstruction = self._forward(target)
        return {'reconstruction': reconstruction,
                'error': tf.reduce_mean(tf.square(reconstruction - target), axis=1)}

    def train_eager(self, signal: tf.Tensor):
        target = self._descriptor((signal - self.signal_mean) / self.signal_std)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self._forward(target) - target))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class SpectralAutoencoderTrainer(AutoencoderTrainer):

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        from ..spectral import DESCRIPTOR_NAMES
        for batch in eval_dataset.take(1):
            out = self.model.eval(*batch)
            target = self.model._descriptor(
                (batch[0] - self.model.signal_mean) / self.model.signal_std)
            fig, ax = plt.subplots(figsize=(9, 4))
            index = np.arange(len(DESCRIPTOR_NAMES))
            ax.bar(index - 0.2, np.asarray(target)[0], 0.4, label='descriptor')
            ax.bar(index + 0.2, np.asarray(out['reconstruction'])[0], 0.4,
                   label='reconstruction')
            ax.set_xticks(index)
            ax.set_xticklabels(DESCRIPTOR_NAMES, rotation=45, ha='right')
            ax.set_ylabel('z-score')
            ax.legend()
            fig.tight_layout()
            fig.savefig(result_dir / 'reconstruction.png')
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break


def get_model(data_root: Path, batch_size: int | None = None) -> SpectralAutoencoder:
    sig_mean, sig_std = autoencoder_norm_params(data_root)
    desc_mean, desc_std = descriptor_norm_params(data_root)
    return SpectralAutoencoder(
        name='dalia_spectral_ae',
        batch_size=batch_size or TrainableAutoencoder.default_batch_size,
        seq_len=BVP_WINDOW,
        signal_mean=sig_mean, signal_std=sig_std,
        descriptor_mean=desc_mean, descriptor_std=desc_std,
    )


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    return SpectralAutoencoderTrainer(get_model(data_root, batch_size), data_root)
