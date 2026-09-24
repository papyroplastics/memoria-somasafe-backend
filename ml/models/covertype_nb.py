from pathlib import Path

import numpy as np
import tensorflow as tf

from ..sources.covertype import N_CLASSES, N_FEATURES
from .common import TrainableModel, Trainer

COUNT_EPS = 1e-3
VAR_SMOOTHING = 1e-3


class CovertypeNB(TrainableModel):
    default_batch_size = 256

    def __init__(self, name: str = 'covertype_nb', batch_size: int | None = None):
        super().__init__(name=name)

        self.batch_size = batch_size or self.default_batch_size
        self.in_shape = (self.batch_size, N_FEATURES)
        self.label_shape = (self.batch_size, N_CLASSES)

        self.class_count = tf.Variable(tf.zeros([N_CLASSES]), name='class_count')
        self.feature_sum = tf.Variable(tf.zeros([N_CLASSES, N_FEATURES]), name='feature_sum')
        self.feature_sumsq = tf.Variable(tf.zeros([N_CLASSES, N_FEATURES]), name='feature_sumsq')

        signature = [tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)]
        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.label_shape, dtype=tf.float32),
        ])

        self._init_save_restore()

    def _log_posterior(self, features: tf.Tensor) -> tf.Tensor:
        count = tf.maximum(self.class_count, COUNT_EPS)
        mean = self.feature_sum / count[:, None]
        raw_var = self.feature_sumsq / count[:, None] - tf.square(mean)

        total_count = tf.maximum(tf.reduce_sum(self.class_count), COUNT_EPS)
        global_mean = tf.reduce_sum(self.feature_sum, axis=0) / total_count
        global_var = tf.reduce_sum(self.feature_sumsq, axis=0) / total_count - tf.square(global_mean)
        var = tf.maximum(raw_var, VAR_SMOOTHING * tf.reduce_max(global_var))

        log_prior = tf.math.log(count / tf.reduce_sum(count))

        diff = features[:, None, :] - mean[None, :, :]
        log_gauss = -0.5 * (tf.math.log(2.0 * np.pi * var) + tf.square(diff) / var)
        return log_prior[None, :] + tf.reduce_sum(log_gauss, axis=-1)

    def eval_eager(self, features: tf.Tensor):
        return {'logits': self._log_posterior(features)}

    def train_eager(self, features: tf.Tensor, labels: tf.Tensor):
        self.class_count.assign_add(tf.reduce_sum(labels, axis=0))
        self.feature_sum.assign_add(tf.matmul(labels, features, transpose_a=True))
        self.feature_sumsq.assign_add(tf.matmul(labels, tf.square(features), transpose_a=True))

        log_posterior = self._log_posterior(features)
        loss = -tf.reduce_mean(tf.reduce_sum(log_posterior * labels, axis=-1))
        return {'loss': loss}


class CovertypeNBTrainer(Trainer):

    primary_metric = 'accuracy'
    dataset_tensors = ['features', 'labels']
    n_eval_inputs = 1
    training_key = 'covertype-wilderness'
    default_holdout = 'last:1'

    def __init__(self, model: CovertypeNB, data_root: Path):
        super().__init__(model, data_root)
        self.model: CovertypeNB = model # type: ignore

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
