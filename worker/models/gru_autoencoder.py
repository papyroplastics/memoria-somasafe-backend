import tensorflow as tf

from .common import Dense, GRUCell, TrainableAutoencoder, AutoencoderTrainer


class GRUAutoencoder(TrainableAutoencoder):
    """GRU autoencoder for reconstruction-based anomaly detection. Same structure
    as ``LSTMAutoencoder`` but with single-state GRU cells, so it is lighter
    (fewer gates / parameters) — a candidate if the LSTM is too heavy for
    on-device training."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 hidden_dim: int, latent_dim: int, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals)

        self.hidden_dim = hidden_dim
        self.encoder_gru = GRUCell(n_signals, hidden_dim)
        self.decoder_gru = GRUCell(hidden_dim, hidden_dim)

        self.to_latent = Dense(hidden_dim, latent_dim, activation=tf.nn.relu)
        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(hidden_dim, n_signals, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal):
        h = self.encoder_gru.zero_state(self.batch_size)
        for t in range(self.seq_len):
            h = self.encoder_gru.step(h, signal[:, t, :])

        dec_hidden = self.latent_to_hidden(self.to_latent(h))

        dh = self.decoder_gru.zero_state(self.batch_size)
        outputs = []
        for _ in range(self.seq_len):
            dh = self.decoder_gru.step(dh, dec_hidden)
            outputs.append(self.output_layer(dh))

        return tf.stack(outputs, axis=1)


def get_trainer(data_root, seed) -> AutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = 12

    model = GRUAutoencoder(
        name='dalia_gru_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, hidden_dim=64, latent_dim=32, learning_rate=1e-3,
    )
    return AutoencoderTrainer(model, window_size=window_size, shift=shift,
                              batch_size=batch_size)
