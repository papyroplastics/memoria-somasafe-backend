import tensorflow as tf

from ..layers import Conv1D, Dense, LSTMCell, sinusoidal_encoding
from ..data import N_COND
from .common import TrainableAutoencoder, AutoencoderTrainer


class LSTMAutoencoder(TrainableAutoencoder):
    """Conditional LSTM autoencoder for reconstruction-based anomaly detection.

    A strided-conv front-end decimates the 8 s @ 64 Hz window (512 steps) down to
    ``reduced_len`` steps before the recurrence, so the LSTM unrolls a short
    sequence instead of 512 (cheaper to train on-device, far better gradients).
    The encoder's final state is fused with the embedded ``cond`` vector and
    projected to the latent (with latent dropout), which seeds the decoder's initial
    state; a fixed sinusoidal positional encoding drives each decoder step
    (content-independent, so it can't collapse to a flat line nor leak signal into
    the score). The reduced-length output is upsampled and smoothed back to the full
    window. Encoder sees ``[BVP, ACC]``; decoder reconstructs BVP only."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 n_cond: int, hidden_dim: int, latent_dim: int, learning_rate: float,
                 cond_embed_dim: int = 16, down_factor: int = 8, pe_dim: int = 16,
                 kernel_size: int = 7, n_outputs: int = 1, diff_weight: float = 1.0,
                 latent_dropout: float = 0.1,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name, batch_size=batch_size, seq_len=seq_len,
                         n_signals=n_signals, n_cond=n_cond, cond_embed_dim=cond_embed_dim,
                         n_outputs=n_outputs, diff_weight=diff_weight,
                         latent_dropout=latent_dropout)
        assert seq_len % down_factor == 0, 'seq_len must be divisible by down_factor'
        self.down_factor = down_factor
        self.reduced_len = seq_len // down_factor
        self.hidden_dim = hidden_dim

        # Encoder: 3 stride-2 convs (downsample x8) -> LSTM -> latent fused with cond.
        self.enc_conv1 = Conv1D(n_signals, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc_conv2 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc_conv3 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.encoder_lstm = LSTMCell(hidden_dim, hidden_dim)
        self.to_latent = Dense(hidden_dim + cond_embed_dim, latent_dim)

        # Decoder: latent seeds the state, positional encoding drives each step,
        # then upsample x8 + smoothing conv back to the full window.
        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.tanh)
        self.pe = sinusoidal_encoding(self.reduced_len, pe_dim)
        self.decoder_lstm = LSTMCell(pe_dim, hidden_dim)
        self.dec_out = Dense(hidden_dim, hidden_dim, activation=tf.nn.relu)
        self.smooth = Conv1D(hidden_dim, n_outputs, kernel_size, activation=None)

        self._bind(learning_rate, beta1, beta2, epsilon)

    def _forward(self, signal, cond, training=False):
        emb = self._embed_cond(cond)
        x = self.enc_conv3(self.enc_conv2(self.enc_conv1(signal)))
        h, c = self.encoder_lstm.zero_state(self.batch_size)
        for t in range(self.reduced_len):
            h, c = self.encoder_lstm.step(h, c, x[:, t, :])
        z = self._drop_latent(self.to_latent(tf.concat([h, emb], axis=1)), training)

        dh = self.latent_to_hidden(z)
        dc = tf.zeros_like(dh)
        pe_dim = self.pe.shape[-1]
        outputs = []
        for t in range(self.reduced_len):
            step_in = tf.broadcast_to(self.pe[t], (self.batch_size, pe_dim))
            dh, dc = self.decoder_lstm.step(dh, dc, step_in)
            outputs.append(self.dec_out(dh))

        seq = tf.stack(outputs, axis=1)
        up = tf.repeat(seq, self.down_factor, axis=1)
        return self.smooth(up)


def get_trainer(batch_size: int | None = None) -> AutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = batch_size or AutoencoderTrainer.default_batch_size

    model = LSTMAutoencoder(
        name='dalia_lstm_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, n_cond=N_COND, hidden_dim=64, latent_dim=16, learning_rate=1e-3,
    )
    return AutoencoderTrainer(model, window_size=window_size, shift=shift,
                              batch_size=batch_size)
