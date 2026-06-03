import tensorflow as tf
from .common import Dense, LSTMCell, TrainableModel
from ..optimizers import Adam
from ..training import mse_loss


class ConditionalLSTMAutoencoder(TrainableModel):
    """LSTM autoencoder for unsupervised cardiovascular anomaly detection.

    Reconstructs a window of ``[BVP, ACC]`` and conditions the latent on a
    static demographics vector plus an activity-context vector. The
    reconstruction error is the anomaly score.

    Exposes the same ``eval``/``train``/``save``/``restore`` signatures as
    ``BasicNN`` so it stays LiteRT-trainable on-device and FedAvg can move
    flattened weights. The optimizer is Adam whose state is kept in
    non-trainable variables (excluded from save/restore).
    """

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 n_signals: int, n_static: int, n_context: int,
                 hidden_dim: int, latent_dim: int, cond_embed_dim: int,
                 learning_rate: float, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals

        self.signal_shape = (batch_size, seq_len, n_signals)
        self.context_shape = (batch_size, n_context)
        self.static_shape = (batch_size, n_static)

        self.cond_dense1 = Dense(n_static + n_context, 32, activation=tf.nn.relu)
        self.cond_dense2 = Dense(32, cond_embed_dim, activation=tf.nn.relu)

        self.hidden_dim = hidden_dim
        self.encoder_lstm = LSTMCell(n_signals, hidden_dim)
        self.decoder_lstm = LSTMCell(hidden_dim, hidden_dim)

        self.fusion = Dense(hidden_dim + cond_embed_dim, latent_dim, activation=tf.nn.relu)
        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(hidden_dim, n_signals, activation=None)

        self.optimizer = Adam(
            self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.context_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.static_shape, dtype=tf.float32),
        ])

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.context_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.static_shape, dtype=tf.float32),
        ])

        self._init_save_restore()


    def _forward(self, signal, context, static):
        cond = self.cond_dense2(self.cond_dense1(tf.concat([context, static], axis=1)))

        h, c = self.encoder_lstm.zero_state(self.batch_size)
        for t in range(self.seq_len):
            h, c = self.encoder_lstm.step(h, c, signal[:, t, :])

        dec_hidden = self.latent_to_hidden(self.fusion(tf.concat([h, cond], axis=1)))

        dh, dc = self.decoder_lstm.zero_state(self.batch_size)
        outputs = []
        for _ in range(self.seq_len):
            dh, dc = self.decoder_lstm.step(dh, dc, dec_hidden)
            outputs.append(self.output_layer(dh))

        return tf.stack(outputs, axis=1)

    def eval_eager(self, signal: tf.Tensor, context: tf.Tensor, static: tf.Tensor):
        reconstruction = self._forward(signal, context, static)
        error = tf.reduce_mean(tf.square(reconstruction - signal), axis=[1, 2])
        return {'reconstruction': reconstruction, 'error': error}

    def train_eager(self, signal: tf.Tensor, context: tf.Tensor, static: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = mse_loss(self._forward(signal, context, static), signal)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}
