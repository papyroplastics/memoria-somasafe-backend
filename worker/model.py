import math
import tensorflow as tf
from .training import mse_loss

class UnboundError(NotImplementedError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

def unbound(*_, **__):
    raise UnboundError('This function is bound dynamically at init time')

class Dense(tf.Module):
    def __init__(self, in_dim, out_dim, activation=tf.tanh):
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.w = tf.Variable(tf.random.uniform(
            shape=[in_dim, out_dim], minval=-limit, maxval=limit
        ))
        self.b = tf.Variable(tf.zeros(shape=[out_dim]))
        self.f = activation

    def __call__(self, data):
        out = data @ self.w + self.b
        return out if self.f is None else self.f(out)

class BasicNN(tf.Module):
    def __init__(self, name: str, batch_size: int,
                 in_dim: int, out_dim: int,
                 hidden_dim: int, hidden_layers: int,
                 learning_rate: float, momentum: float):
        super().__init__(name=name)

        self.batch_size = batch_size

        self.in_shape = (batch_size, in_dim)
        self.out_shape = (batch_size, out_dim)

        self.in_layer = Dense(in_dim, hidden_dim)
        self.hidden_layers = [Dense(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        self.out_layer = Dense(hidden_dim, out_dim, activation=None)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)
        ])

        self.learning_rate = tf.constant(learning_rate)
        self.momentum = tf.constant(momentum)
        self.velocity = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.out_shape, dtype=tf.float32),
        ])

        self.parameter_sizes = [
            int(var.shape.num_elements()) for var in self.trainable_variables
        ]
        self.total_parameter_size = sum(self.parameter_sizes)

        self.save = tf.function(self.save_eager, input_signature=[])
        self.restore = tf.function(self.restore_eager, input_signature=[
            tf.TensorSpec(shape=(self.total_parameter_size,), dtype=tf.float32),
        ])

    def _evaluate_model(self, data):
        activation = self.in_layer(data)

        for layer in self.hidden_layers:
            activation = layer(activation)

        return self.out_layer(activation)

    def eval_eager(self, data: tf.Tensor):
        return {
            'result': self._evaluate_model(data)
        }

    eval: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def train_eager(self, data: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            prediction = self._evaluate_model(data)
            loss = mse_loss(prediction, labels)

        grads = tape.gradient(loss, self.trainable_variables)

        for i, var in enumerate(self.trainable_variables):
            self.velocity[i].assign(self.momentum * self.velocity[i] + grads[i]) # type: ignore
            var.assign_sub(self.learning_rate * self.velocity[i])

        return {
            'loss': loss
        }

    train: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def save_eager(self):
        return {
            'parameters': tf.concat([
                tf.reshape(var, (-1,)) for var in self.trainable_variables
            ], axis=0)
        }

    save: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def restore_eager(self, parameters: tf.Tensor):
        idx = 0
        for i, var in enumerate(self.trainable_variables):
            size = self.parameter_sizes[i]
            var.assign(tf.reshape(parameters[idx:idx + size], var.shape))
            idx += size

        return {
            'parameter_count': tf.constant(self.total_parameter_size, dtype=tf.int32)
        }

    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore


class LSTMCell(tf.Module):
    """Minimal LSTM cell built from plain ``tf.Variable``s.

    Implemented by hand (rather than ``tf.keras.layers.LSTM``) so its weights are
    real ``tf.Variable``s tracked by ``tf.Module.trainable_variables`` - Keras 3
    layers wrap their weights in ``keras.Variable`` and would not be collected,
    breaking the flatten-based save/restore and FedAvg transfer this project
    relies on. Sequences are unrolled statically by the caller, which also keeps
    the exported graph control-flow-free and friendlier to TFLite conversion.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim

        limit_w = math.sqrt(6.0 / (in_dim + 4 * hidden_dim))
        self.W = tf.Variable(tf.random.uniform(
            shape=[in_dim, 4 * hidden_dim], minval=-limit_w, maxval=limit_w))

        limit_u = math.sqrt(6.0 / (hidden_dim + 4 * hidden_dim))
        self.U = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, 4 * hidden_dim], minval=-limit_u, maxval=limit_u))

        self.b = tf.Variable(tf.zeros(shape=[4 * hidden_dim]))

    def zero_state(self, batch_size: int):
        h = tf.zeros([batch_size, self.hidden_dim])
        c = tf.zeros([batch_size, self.hidden_dim])
        return h, c

    def step(self, h, c, x_t):
        z = x_t @ self.W + h @ self.U + self.b
        i, f, g, o = tf.split(z, 4, axis=-1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        c_new = f * c + i * g
        h_new = o * tf.tanh(c_new)
        return h_new, c_new


class ConditionalLSTMAutoencoder(tf.Module):
    """LSTM autoencoder for unsupervised cardiovascular anomaly detection.

    Ported from the legacy PyTorch ``ConditionalLSTMAutoencoder``. The model
    reconstructs a window of ``[BVP, ACC]`` and conditions the latent on a
    static demographics vector plus an activity-context vector. The
    reconstruction error is the anomaly score.

    Exposes the same ``eval``/``train``/``save``/``restore`` signatures as
    ``BasicNN`` so it stays LiteRT-trainable on-device and FedAvg can move
    flattened weights. The optimizer is a hand-implemented Adam whose state is
    kept in non-trainable variables (and therefore excluded from save/restore).
    """

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 n_signals: int = 2, n_static: int = 6, n_context: int = 2,
                 hidden_dim: int = 64, latent_dim: int = 32, cond_embed_dim: int = 16,
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals

        self.signal_shape = (batch_size, seq_len, n_signals)
        self.context_shape = (batch_size, n_context)
        self.static_shape = (batch_size, n_static)

        # Conditioning branch: fuse static demographics + activity context.
        self.cond_dense1 = Dense(n_static + n_context, 32, activation=tf.nn.relu)
        self.cond_dense2 = Dense(32, cond_embed_dim, activation=tf.nn.relu)

        self.hidden_dim = hidden_dim

        # Recurrent encoder / decoder (hand-rolled cells, unrolled below).
        self.encoder_lstm = LSTMCell(n_signals, hidden_dim)
        self.decoder_lstm = LSTMCell(hidden_dim, hidden_dim)

        # Fusion of the time-series latent with the conditioning embedding, and
        # projections in/out of the bottleneck.
        self.fusion = Dense(hidden_dim + cond_embed_dim, latent_dim, activation=tf.nn.relu)
        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(hidden_dim, n_signals, activation=None)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.context_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.static_shape, dtype=tf.float32),
        ])

        # Adam optimizer state (non-trainable, local to the device).
        self.learning_rate = tf.constant(learning_rate)
        self.beta1 = tf.constant(beta1)
        self.beta2 = tf.constant(beta2)
        self.epsilon = tf.constant(epsilon)
        self.adam_m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.trainable_variables]
        self.adam_v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in self.trainable_variables]
        self.adam_step = tf.Variable(0.0, trainable=False)

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.context_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.static_shape, dtype=tf.float32),
        ])

        self.parameter_sizes = [
            int(var.shape.num_elements()) for var in self.trainable_variables
        ]
        self.total_parameter_size = sum(self.parameter_sizes)

        self.save = tf.function(self.save_eager, input_signature=[])
        self.restore = tf.function(self.restore_eager, input_signature=[
            tf.TensorSpec(shape=(self.total_parameter_size,), dtype=tf.float32),
        ])

    def _evaluate_model(self, signal, context, static):
        batch_size = tf.shape(signal)[0]

        cond = self.cond_dense1(tf.concat([context, static], axis=1))
        cond = self.cond_dense2(cond)

        # Encode the signal window; keep the final hidden state (statically
        # unrolled over the fixed window length).
        h, c = self.encoder_lstm.zero_state(batch_size)
        for t in range(self.seq_len):
            h, c = self.encoder_lstm.step(h, c, signal[:, t, :])
        h_n = h

        latent = self.fusion(tf.concat([h_n, cond], axis=1))
        dec_hidden = self.latent_to_hidden(latent)

        # Decode: feed the bottleneck representation at every timestep, collect
        # the per-step hidden states, and project each to the signal space.
        dh, dc = self.decoder_lstm.zero_state(batch_size)
        outputs = []
        for _ in range(self.seq_len):
            dh, dc = self.decoder_lstm.step(dh, dc, dec_hidden)
            outputs.append(self.output_layer(dh))

        return tf.stack(outputs, axis=1)

    def eval_eager(self, signal: tf.Tensor, context: tf.Tensor, static: tf.Tensor):
        reconstruction = self._evaluate_model(signal, context, static)
        error = tf.reduce_mean(tf.square(reconstruction - signal), axis=[1, 2])
        return {
            'reconstruction': reconstruction,
            'error': error,
        }

    eval: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def train_eager(self, signal: tf.Tensor, context: tf.Tensor, static: tf.Tensor):
        with tf.GradientTape() as tape:
            reconstruction = self._evaluate_model(signal, context, static)
            loss = mse_loss(reconstruction, signal)

        grads = tape.gradient(loss, self.trainable_variables)

        self.adam_step.assign_add(1.0)
        t = self.adam_step
        lr_t = self.learning_rate * tf.sqrt(1.0 - tf.pow(self.beta2, t)) / (1.0 - tf.pow(self.beta1, t))

        for i, var in enumerate(self.trainable_variables):
            g = grads[i]
            self.adam_m[i].assign(self.beta1 * self.adam_m[i] + (1.0 - self.beta1) * g)
            self.adam_v[i].assign(self.beta2 * self.adam_v[i] + (1.0 - self.beta2) * tf.square(g))
            var.assign_sub(lr_t * self.adam_m[i] / (tf.sqrt(self.adam_v[i]) + self.epsilon))

        return {
            'loss': loss
        }

    train: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def save_eager(self):
        return {
            'parameters': tf.concat([
                tf.reshape(var, (-1,)) for var in self.trainable_variables
            ], axis=0)
        }

    save: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def restore_eager(self, parameters: tf.Tensor):
        idx = 0
        for i, var in enumerate(self.trainable_variables):
            size = self.parameter_sizes[i]
            var.assign(tf.reshape(parameters[idx:idx + size], var.shape))
            idx += size

        return {
            'parameter_count': tf.constant(self.total_parameter_size, dtype=tf.int32)
        }

    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore
