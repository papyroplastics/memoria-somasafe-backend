from pathlib import Path

import tensorflow as tf

from ..layers import Dense, GRUCell
from ..preprocessing import BVP_WINDOW
from .common import SignalAutoencoder, AutoencoderTrainer


class GRUAutoencoder(SignalAutoencoder):
    def __init__(self, name: str, batch_size: int, seq_len: int,
                 n_signals: int = 1, hidden_dim: int = 64, latent_dim: int = 32,
                 learning_rate: float = 1e-3, n_outputs: int = 1,
                 diff_weight: float = 1.0, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_outputs=n_outputs,
                         diff_weight=diff_weight)

        self.enc_gru1 = GRUCell(n_signals, hidden_dim)
        self.enc_gru2 = GRUCell(hidden_dim, latent_dim)
        self.to_latent = Dense(latent_dim, latent_dim)

        self.dec_gru1 = GRUCell(latent_dim, hidden_dim)
        self.dec_gru2 = GRUCell(hidden_dim, hidden_dim)
        self.out_dense = Dense(hidden_dim, n_outputs)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal):
        h1 = self.enc_gru1.zero_state(self.batch_size)
        h2 = self.enc_gru2.zero_state(self.batch_size)
        for t in range(self.seq_len):
            h1 = self.enc_gru1.step(h1, signal[:, t, :])
            h2 = self.enc_gru2.step(h2, h1)

        z = self.to_latent(h2)

        dh1 = self.dec_gru1.zero_state(self.batch_size)
        dh2 = self.dec_gru2.zero_state(self.batch_size)
        outputs = []
        for t in range(self.seq_len):
            dh1 = self.dec_gru1.step(dh1, z)
            dh2 = self.dec_gru2.step(dh2, dh1)
            outputs.append(self.out_dense(dh2))

        return tf.stack(outputs, axis=1)


def get_model(data_root: Path, batch_size: int | None = None) -> GRUAutoencoder:
    return GRUAutoencoder(
        name='dalia_gru_ae',
        batch_size=batch_size or SignalAutoencoder.default_batch_size,
        seq_len=BVP_WINDOW,
    )


def get_trainer(data_root: Path, batch_size: int | None = None) -> AutoencoderTrainer:
    return AutoencoderTrainer(get_model(data_root, batch_size), data_root)
