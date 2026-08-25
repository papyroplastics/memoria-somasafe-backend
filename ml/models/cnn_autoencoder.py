from pathlib import Path

import tensorflow as tf

from ..layers import Conv1D, Dense, relu, upsample2
from ..preprocessing import BVP_WINDOW
from .common import TrainableAutoencoder, AutoencoderTrainer, autoencoder_norm_params

class CNNAutoencoder(TrainableAutoencoder):
    """Conv1D autoencoder over an 8-second BVP window, scored by reconstruction error.

    Strided convolutions downsample the window, a dense projection collapses the
    remaining time axis into one ``latent_dim`` code, and the decoder projects back and
    reconstructs with nearest-neighbour upsampling + convolutions. Collapsing time is
    the point: strided convs alone only shrink the time axis, leaving
    ``(seq_len / 8) * channels`` values — as many numbers as the input, enough to copy
    it verbatim rather than learn anything. No recurrence, so it quantizes cleanly and
    avoids the LSTM's 1024-step unroll. ``seq_len`` must be divisible by ``2 ** 3``."""

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 signal_mean, signal_std, n_signals: int = 1,
                 hidden_dim: int = 32, latent_dim: int = 32,
                 kernel_size: int = 5, n_outputs: int = 1,
                 diff_weight: float = 1.0, learning_rate: float = 5e-4,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_outputs=n_outputs,
                         diff_weight=diff_weight,
                         signal_mean=signal_mean, signal_std=signal_std)

        self.conv_blocks = 4
        self.subsample_factor = 2 ** self.conv_blocks

        self.enc_in = Conv1D(n_signals, hidden_dim, kernel_size, activation=relu)
        self.enc_layers = []
        for i in range(self.conv_blocks):
            self.enc_layers.append(
                Conv1D(hidden_dim, hidden_dim, kernel_size, activation=relu))
            self.enc_layers.append(
                Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=relu))

        self.enc_steps = seq_len // self.subsample_factor
        self.enc_channels = hidden_dim
        self.enc_flat = self.enc_steps * hidden_dim
        self.to_latent = Dense(self.enc_flat, latent_dim)
        self.from_latent = Dense(latent_dim, self.enc_flat)

        self.dec_layers = []
        self.dec_out = Conv1D(hidden_dim, n_outputs, kernel_size, activation=None)
        
        for i in range(self.conv_blocks):
            self.dec_layers.append(
                Conv1D(hidden_dim, hidden_dim, kernel_size, activation=relu))
            self.dec_layers.append(
                Conv1D(hidden_dim, hidden_dim, kernel_size, activation=relu))
            self.dec_layers.append(upsample2)


        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal):
        x = self.enc_in(signal)
        for enc in self.enc_layers:
            x = enc(x)

        flat = tf.reshape(x, [-1, self.enc_flat])
        z = self.to_latent(flat)

        x = tf.reshape(self.from_latent(z), [-1, self.enc_steps, self.enc_channels])

        for dec in self.dec_layers:
            x = dec(x)

        x = self.dec_out(x)
        return x


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    sig_mean, sig_std = autoencoder_norm_params(data_root)
    model = CNNAutoencoder(
        name='dalia_cnn_ae', batch_size=batch_size or TrainableAutoencoder.default_batch_size,
        seq_len=BVP_WINDOW,
        signal_mean=sig_mean, signal_std=sig_std,
    )
    return AutoencoderTrainer(model)
