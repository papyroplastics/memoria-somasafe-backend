import numpy as np
import tensorflow as tf

from ..layers import Conv1D, Dense, LSTMCell, sinusoidal_encoding
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..data import windowed_normalized
from .common import TrainableModel, AutoencoderTrainer
from ..optimizers import Adam


class ConditionalLSTMAutoencoder(TrainableModel):
    """LSTM autoencoder that conditions the latent on a generic ``cond`` vector.

    Architecturally mirrors ``LSTMAutoencoder`` (decimating conv front-end,
    positional-encoding decoder, BVP-only reconstruction) but fuses an embedded
    conditioning vector into the latent. ``cond`` is a single concatenated vector
    (currently the demographics ``static`` vector) so extra conditioning can be
    added later without changing the signature. Kept separate from
    ``TrainableAutoencoder`` so the value of the conditioning can be measured
    against the plain ``LSTMAutoencoder``."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 n_cond: int, hidden_dim: int, latent_dim: int, cond_embed_dim: int,
                 learning_rate: float, down_factor: int = 8, pe_dim: int = 16,
                 kernel_size: int = 7, n_outputs: int = 1, diff_weight: float = 1.0,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)
        assert seq_len % down_factor == 0, 'seq_len must be divisible by down_factor'

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.n_outputs = n_outputs
        self.diff_weight = diff_weight
        self.down_factor = down_factor
        self.reduced_len = seq_len // down_factor
        self.hidden_dim = hidden_dim

        self.signal_shape = (batch_size, seq_len, n_signals)
        self.cond_shape = (batch_size, n_cond)

        self.cond_dense1 = Dense(n_cond, 32, activation=tf.nn.relu)
        self.cond_dense2 = Dense(32, cond_embed_dim, activation=tf.nn.relu)

        self.enc_conv1 = Conv1D(n_signals, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc_conv2 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.enc_conv3 = Conv1D(hidden_dim, hidden_dim, kernel_size, stride=2, activation=tf.nn.relu)
        self.encoder_lstm = LSTMCell(hidden_dim, hidden_dim)
        self.to_latent = Dense(hidden_dim + cond_embed_dim, latent_dim)

        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.tanh)
        self.pe = sinusoidal_encoding(self.reduced_len, pe_dim)
        self.decoder_lstm = LSTMCell(pe_dim, hidden_dim)
        self.dec_out = Dense(hidden_dim, hidden_dim, activation=tf.nn.relu)
        self.smooth = Conv1D(hidden_dim, n_outputs, kernel_size, activation=None)

        self.optimizer = Adam(
            self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.cond_shape, dtype=tf.float32),
        ])
        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.cond_shape, dtype=tf.float32),
        ])

        self._init_save_restore()

    def _forward(self, signal, cond):
        cond_embed = self.cond_dense2(self.cond_dense1(cond))

        x = self.enc_conv3(self.enc_conv2(self.enc_conv1(signal)))
        h, c = self.encoder_lstm.zero_state(self.batch_size)
        for t in range(self.reduced_len):
            h, c = self.encoder_lstm.step(h, c, x[:, t, :])
        z = self.to_latent(tf.concat([h, cond_embed], axis=1))

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

    def eval_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        reconstruction = self._forward(signal, cond)
        target = signal[..., :self.n_outputs]
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, target)}

    def train_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        target = signal[..., :self.n_outputs]
        with tf.GradientTape() as tape:
            reconstruction = self._forward(signal, cond)
            loss = (mse_loss(reconstruction, target)
                    + self.diff_weight * first_difference_loss(reconstruction, target))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class ConditionalAutoencoderTrainer(AutoencoderTrainer):
    """Adds the per-subject conditioning vector (the normalized ``static``
    demographics) to the windowed signal. Reuses the reconstruction metrics from
    ``AutoencoderTrainer``."""

    def _windowed(self, subject_dir):
        sig_ds, count = windowed_normalized(
            subject_dir.parent, subject_dir.name, self.window_size, self.shift)
        static = np.load(subject_dir / 'static.npy').astype(np.float32)
        cond_ds = tf.data.Dataset.from_tensors(static).repeat()
        return tf.data.Dataset.zip((sig_ds, cond_ds)), count

    def representative_dataset(self, dataset):
        return dataset.take(10).map(lambda s, c: {'signal': s, 'cond': c})


def get_trainer(data_root, seed, batch_size=None) -> ConditionalAutoencoderTrainer:
    sample_rate = 64
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    batch_size = batch_size or ConditionalAutoencoderTrainer.default_batch_size

    model = ConditionalLSTMAutoencoder(
        name='dalia_cond_lstm_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, n_cond=6, hidden_dim=64, latent_dim=32, cond_embed_dim=16,
        learning_rate=1e-3,
    )
    return ConditionalAutoencoderTrainer(model, window_size=window_size,
                                         shift=shift, batch_size=batch_size)
