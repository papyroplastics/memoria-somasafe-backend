from pathlib import Path

import numpy as np
import tensorflow as tf

from ..layers import Dense, relu, softmax_cross_entropy
from .common import BackpropModel, Trainer

N_PIXELS = 784
N_CLASSES = 10


class MnistMLP(BackpropModel):
    default_batch_size = 32

    def __init__(self, name: str = 'mnist_mlp', batch_size: int | None = None,
                 hidden1: int = 128, hidden2: int = 64,
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size or self.default_batch_size
        self.in_shape = (self.batch_size, N_PIXELS)
        self.label_shape = (self.batch_size, N_CLASSES)

        self.in_layer = Dense(N_PIXELS, hidden1, activation=relu)
        self.hidden_layer = Dense(hidden1, hidden2, activation=relu)
        self.out_layer = Dense(hidden2, N_CLASSES)

        self._init_optimizer(learning_rate, beta1, beta2, epsilon)

        signature = [tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)]
        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.label_shape, dtype=tf.float32),
        ])

        self._init_save_restore()

    def _logits(self, images):
        activation = self.in_layer(images)
        activation = self.hidden_layer(activation)
        return self.out_layer(activation)

    def eval_eager(self, images: tf.Tensor):
        return {'logits': self._logits(images)}

    def train_eager(self, images: tf.Tensor, labels: tf.Tensor):
        return self._apply_step(lambda: tf.reduce_mean(
            softmax_cross_entropy(labels, self._logits(images))))


class MnistMLPTrainer(Trainer):

    primary_metric = 'accuracy'
    dataset_tensors = ['images', 'labels']
    n_eval_inputs = 1
    training_key = 'mnist-iid'
    calibration_key = 'mnist-test'

    def __init__(self, model: MnistMLP, data_root: Path):
        super().__init__(model, data_root)
        self.model: MnistMLP = model # type: ignore

    def subject_arrays(self, sid):
        labels = self.data.labels(sid)
        assert labels is not None
        return (self.data.datapoints(sid), labels)

    def eval_metrics(self, datapoints, outputs):
        correct, total = 0.0, 0.0
        for (_, y), out in zip(datapoints, outputs):
            pred = np.argmax(np.asarray(out['logits']), axis=-1)
            truth = np.argmax(np.asarray(y), axis=-1)
            correct += float(np.sum(pred == truth))
            total += float(len(truth))
        return {'accuracy': correct / total if total else 0.0}
