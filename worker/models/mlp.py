import tensorflow as tf
from .common import Dense, TrainableModel
from ..optimizers import MomentumSGD
from ..training import mse_loss


class MLP(TrainableModel):
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

        self._init_save_restore()

        self.optimizer = MomentumSGD(self.trainable_variables, learning_rate, momentum)

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.out_shape, dtype=tf.float32),
        ])

    def _forward(self, data):
        activation = self.in_layer(data)
        for layer in self.hidden_layers:
            activation = layer(activation)
        return self.out_layer(activation)

    def eval_eager(self, data: tf.Tensor):
        return {'result': self._forward(data)}

    def train_eager(self, data: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = mse_loss(self._forward(data), labels)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}
