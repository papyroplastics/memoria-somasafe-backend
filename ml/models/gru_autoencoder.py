import tensorflow as tf

from ..layers import Conv1D, Dense, GRUCell, sinusoidal_encoding
from .common import TrainableAutoencoder, AutoencoderTrainer


class GRUAutoencoder(TrainableAutoencoder):
    """GRU autoencoder for reconstruction-based anomaly detection. Same decimating
    conv front-end + positional-encoding decoder as ``LSTMAutoencoder`` but with
    single-state GRU cells, so it is lighter (fewer gates / parameters) — a
    candidate if the LSTM is too heavy for on-device training."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 hidden_dim: int, latent_dim: int, learning_rate: float,
                 down_factor: int = 8, pe_dim: int = 16, kernel_size: int = 7,
                 n_outputs: int = 1, diff_weight: float = 1.0,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_outputs=n_outputs, diff_weight=diff_weight)
        assert seq_len % down_factor == 0, 'seq_len must be divisible by down_factor'
        self.down_factor = down_factor
        self.reduced_len = seq_len // down_factor
        self.hidden_dim = hidden_dim

        self.enc_conv1 = Conv1D(n_signals, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc_conv2 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc_conv3 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.encoder_gru = GRUCell(hidden_dim, hidden_dim)
        self.to_latent = Dense(hidden_dim, latent_dim)

        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.tanh)
        self.pe = sinusoidal_encoding(self.reduced_len, pe_dim)
        self.decoder_gru = GRUCell(pe_dim, hidden_dim)
        self.dec_out = Dense(hidden_dim, hidden_dim, activation=tf.nn.relu)
        self.smooth = Conv1D(hidden_dim, n_outputs, kernel_size, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal):
        x = self.enc_conv3(self.enc_conv2(self.enc_conv1(signal)))
        h = self.encoder_gru.zero_state(self.batch_size)
        for t in range(self.reduced_len):
            h = self.encoder_gru.step(h, x[:, t, :])
        z = self.to_latent(h)

        dh = self.latent_to_hidden(z)
        pe_dim = self.pe.shape[-1]
        outputs = []
        for t in range(self.reduced_len):
            step_in = tf.broadcast_to(self.pe[t], (self.batch_size, pe_dim))
            dh = self.decoder_gru.step(dh, step_in)
            outputs.append(self.dec_out(dh))

        seq = tf.stack(outputs, axis=1)
        up = tf.repeat(seq, self.down_factor, axis=1)
        return self.smooth(up)


def get_trainer(data_root, seed, batch_size=None) -> AutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = batch_size or AutoencoderTrainer.default_batch_size

    model = GRUAutoencoder(
        name='dalia_gru_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, hidden_dim=64, latent_dim=32, learning_rate=1e-3,
    )
    return AutoencoderTrainer(model, window_size=window_size, shift=shift,
                              batch_size=batch_size)
