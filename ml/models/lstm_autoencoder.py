from pathlib import Path

import tensorflow as tf

from ..layers import Dense, LSTMCell
from ..preprocessing import N_COND, BVP_WINDOW
from .common import TrainableAutoencoder, AutoencoderTrainer, autoencoder_norm_params


class LSTMAutoencoder(TrainableAutoencoder):
    """Conditional LSTM autoencoder for reconstruction-based anomaly detection.

    Two stacked LSTMCells encode the full-length signal to a latent vector,
    which is fused with the embedded condition and then fed at every decoder
    step to drive two stacked LSTMCells back to the original length."""

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 signal_mean, signal_std, cond_mean, cond_std, n_signals: int = 2,
                 n_cond: int = N_COND, hidden_dim: int = 64, latent_dim: int = 32,
                 learning_rate: float = 1e-3, cond_embed_dim: int = 16, n_outputs: int = 1,
                 diff_weight: float = 1.0, latent_dropout: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_cond=n_cond, cond_embed_dim=cond_embed_dim,
                         n_outputs=n_outputs, diff_weight=diff_weight,
                         latent_dropout=latent_dropout,
                         signal_mean=signal_mean, signal_std=signal_std,
                         cond_mean=cond_mean, cond_std=cond_std)

        self.enc_lstm1 = LSTMCell(n_signals, hidden_dim)
        self.enc_lstm2 = LSTMCell(hidden_dim, latent_dim)
        self.to_latent = Dense(latent_dim + cond_embed_dim, latent_dim)

        self.dec_lstm1 = LSTMCell(latent_dim, hidden_dim)
        self.dec_lstm2 = LSTMCell(hidden_dim, hidden_dim)
        self.out_dense = Dense(hidden_dim, n_outputs)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal, cond, training=False):
        emb = self._embed_cond(cond)

        h1, c1 = self.enc_lstm1.zero_state(self.batch_size)
        h2, c2 = self.enc_lstm2.zero_state(self.batch_size)
        for t in range(self.seq_len):
            h1, c1 = self.enc_lstm1.step(h1, c1, signal[:, t, :])
            h2, c2 = self.enc_lstm2.step(h2, c2, h1)

        z = self._drop_latent(self.to_latent(tf.concat([h2, emb], axis=1)), training)

        dh1, dc1 = self.dec_lstm1.zero_state(self.batch_size)
        dh2, dc2 = self.dec_lstm2.zero_state(self.batch_size)
        outputs = []
        for t in range(self.seq_len):
            dh1, dc1 = self.dec_lstm1.step(dh1, dc1, z)
            dh2, dc2 = self.dec_lstm2.step(dh2, dc2, dh1)
            outputs.append(self.out_dense(dh2))

        return tf.stack(outputs, axis=1)


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    sig_mean, sig_std, cond_mean, cond_std = autoencoder_norm_params(data_root)
    model = LSTMAutoencoder(
        name='dalia_lstm_ae', batch_size=batch_size or TrainableAutoencoder.default_batch_size,
        seq_len=BVP_WINDOW,
        signal_mean=sig_mean, signal_std=sig_std,
        cond_mean=cond_mean, cond_std=cond_std,
    )
    return AutoencoderTrainer(model)
