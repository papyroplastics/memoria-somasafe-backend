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
