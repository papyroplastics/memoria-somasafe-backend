import tensorflow as tf

from .common import Dense, LSTMCell, TrainableAutoencoder, AutoencoderTrainer


class LSTMAutoencoder(TrainableAutoencoder):
    """Non-conditional LSTM autoencoder for reconstruction-based anomaly
    detection. Encodes a window of ``[BVP, ACC]`` to a latent vector and decodes
    it back; the reconstruction error is the anomaly score. Mirrors
    ``ConditionalLSTMAutoencoder`` without the demographics/context fusion so the
    two can be compared."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 hidden_dim: int, latent_dim: int, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals)

        self.hidden_dim = hidden_dim
        self.encoder_lstm = LSTMCell(n_signals, hidden_dim)
        self.decoder_lstm = LSTMCell(hidden_dim, hidden_dim)

        self.to_latent = Dense(hidden_dim, latent_dim, activation=tf.nn.relu)
        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(hidden_dim, n_signals, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal):
        h, c = self.encoder_lstm.zero_state(self.batch_size)
        for t in range(self.seq_len):
            h, c = self.encoder_lstm.step(h, c, signal[:, t, :])

        dec_hidden = self.latent_to_hidden(self.to_latent(h))

        dh, dc = self.decoder_lstm.zero_state(self.batch_size)
        outputs = []
        for _ in range(self.seq_len):
            dh, dc = self.decoder_lstm.step(dh, dc, dec_hidden)
            outputs.append(self.output_layer(dh))

        return tf.stack(outputs, axis=1)


def get_trainer(data_root, seed, batch_size=None) -> AutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = batch_size or AutoencoderTrainer.default_batch_size

    model = LSTMAutoencoder(
        name='dalia_lstm_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, hidden_dim=64, latent_dim=32, learning_rate=1e-3,
    )
    return AutoencoderTrainer(model, window_size=window_size, shift=shift,
                              batch_size=batch_size)
