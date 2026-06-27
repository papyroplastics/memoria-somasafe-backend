import tensorflow as tf

from ..layers import Conv1D, FiLM
from ..data import N_COND
from .common import TrainableAutoencoder, AutoencoderTrainer


class CNNAutoencoder(TrainableAutoencoder):
    """Conditional Conv1D autoencoder for reconstruction-based anomaly detection.

    Strided convolutions downsample the window to a temporal bottleneck; nearest-
    neighbour upsampling + convolutions reconstruct it. No recurrence, so it
    quantizes cleanly for TFLM and avoids the LSTM's 1024-step unroll. The decoder
    is conditioned with FiLM at every layer (per-channel scale/shift from the
    embedded ``cond`` vector), and the bottleneck is deliberately small with latent
    dropout, so the decoder leans on the condition to generate the *expected normal*
    signal rather than copying the input — which is what lets reconstruction error
    separate anomalies it would otherwise reproduce. Encoder sees ``[BVP, ACC]``;
    decoder reconstructs BVP only. ``seq_len`` must be divisible by ``2 ** 3``."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 n_cond: int, hidden_dim: int, latent_dim: int, cond_embed_dim: int = 16,
                 kernel_size: int = 7, n_outputs: int = 1, diff_weight: float = 1.0,
                 latent_dropout: float = 0.1, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_cond=n_cond, cond_embed_dim=cond_embed_dim,
                         n_outputs=n_outputs, diff_weight=diff_weight,
                         latent_dropout=latent_dropout)

        self.enc1 = Conv1D(n_signals, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc2 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc3 = Conv1D(hidden_dim, latent_dim, kernel_size, stride=2, activation=tf.nn.relu)

        self.dec1 = Conv1D(latent_dim, hidden_dim, kernel_size, activation=tf.nn.relu)
        self.film1 = FiLM(cond_embed_dim, hidden_dim)
        self.dec2 = Conv1D(hidden_dim, hidden_dim, kernel_size, activation=tf.nn.relu)
        self.film2 = FiLM(cond_embed_dim, hidden_dim)
        self.dec3 = Conv1D(hidden_dim, n_outputs, kernel_size, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal, cond, training=False):
        emb = self._embed_cond(cond)
        x = self.enc3(self.enc2(self.enc1(signal)))
        x = self._drop_latent(x, training)
        x = self.film1(self.dec1(tf.repeat(x, 2, axis=1)), emb)
        x = self.film2(self.dec2(tf.repeat(x, 2, axis=1)), emb)
        x = self.dec3(tf.repeat(x, 2, axis=1))
        return x


def get_trainer(data_root, seed, batch_size=None) -> AutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = batch_size or AutoencoderTrainer.default_batch_size

    model = CNNAutoencoder(
        name='dalia_cnn_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, n_cond=N_COND, hidden_dim=32, latent_dim=8, learning_rate=1e-3,
    )
    return AutoencoderTrainer(model, window_size=window_size, shift=shift,
                              batch_size=batch_size)
