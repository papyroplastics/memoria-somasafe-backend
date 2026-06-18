import tensorflow as tf

from .common import Conv1D, TrainableAutoencoder, AutoencoderTrainer


class CNNAutoencoder(TrainableAutoencoder):
    """Conv1D autoencoder for reconstruction-based anomaly detection.

    Strided convolutions downsample the window to a temporal bottleneck;
    nearest-neighbour upsampling + convolutions reconstruct it. No recurrence,
    so it quantizes cleanly for TFLM and avoids the LSTM's 1024-step unroll.
    ``seq_len`` must be divisible by ``2 ** 3`` (three stride-2 stages)."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 hidden_dim: int, latent_dim: int, kernel_size: int = 7,
                 learning_rate: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals)

        self.enc1 = Conv1D(n_signals, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc2 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc3 = Conv1D(hidden_dim, latent_dim, kernel_size, stride=2, activation=tf.nn.relu)

        self.dec1 = Conv1D(latent_dim, hidden_dim, kernel_size, activation=tf.nn.relu)
        self.dec2 = Conv1D(hidden_dim, hidden_dim, kernel_size, activation=tf.nn.relu)
        self.dec3 = Conv1D(hidden_dim, n_signals, kernel_size, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal):
        x = self.enc3(self.enc2(self.enc1(signal)))
        x = self.dec1(tf.repeat(x, 2, axis=1))
        x = self.dec2(tf.repeat(x, 2, axis=1))
        x = self.dec3(tf.repeat(x, 2, axis=1))
        return x


def get_trainer(data_root, seed) -> AutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = 12

    model = CNNAutoencoder(
        name='dalia_cnn_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, hidden_dim=32, latent_dim=16, learning_rate=1e-3,
    )
    return AutoencoderTrainer(model, window_size=window_size, shift=shift,
                              batch_size=batch_size)
