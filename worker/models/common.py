import math
import tensorflow as tf

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


class LSTMCell(tf.Module):
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


class TrainableModel(tf.Module):
    """Base class for all LiteRT-trainable / FedAvg-compatible models.

    Subclasses must:
      1. Create all trainable layers/variables.
      2. Bind ``self.eval`` and ``self.train`` as ``tf.function``s with the
         appropriate ``input_signature``.
      3. Call ``self._init_save_restore()`` once all trainable variables exist
         (optimizer state is non-trainable and need not exist yet).
    """

    eval: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    train: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    save: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def _init_save_restore(self):
        self.parameter_sizes = [
            int(var.shape.num_elements()) for var in self.trainable_variables
        ]
        self.total_parameter_size = sum(self.parameter_sizes)
        self.save = tf.function(self.save_eager, input_signature=[])
        self.restore = tf.function(self.restore_eager, input_signature=[
            tf.TensorSpec(shape=(self.total_parameter_size,), dtype=tf.float32),
        ])

    def save_eager(self):
        return {
            'parameters': tf.concat([
                tf.reshape(var, (-1,)) for var in self.trainable_variables
            ], axis=0)
        }

    def restore_eager(self, parameters: tf.Tensor):
        idx = 0
        for i, var in enumerate(self.trainable_variables):
            size = self.parameter_sizes[i]
            var.assign(tf.reshape(parameters[idx:idx + size], var.shape))
            idx += size
        return {
            'parameter_count': tf.constant(self.total_parameter_size, dtype=tf.int32)
        }
