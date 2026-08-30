from pathlib import Path

import numpy as np
import tensorflow as tf

from ..layers import Dense, relu
from .common import TrainableModel, Trainer
from ..preprocessing import N_FEATURES
from ..sources.dalia import MIXED
from ..optimizers import Adam


class FeatureMLP(TrainableModel):
    """Supervised binary anomaly classifier over hand-crafted window features. """

    default_batch_size = 1

    def __init__(self, name: str, batch_size: int,
                 n_features: int = N_FEATURES,
                 hidden_dim: int = 32, hidden_layers: int = 3, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.in_shape = (batch_size, n_features)
        self.label_shape = (batch_size, 1)

        self.in_layer = Dense(n_features, hidden_dim, activation=relu)
        self.out_layer = Dense(hidden_dim, 1)
        self.hidden_layers = [
            Dense(hidden_dim, hidden_dim, activation=relu) for _ in range(hidden_layers)
        ]

        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        # Features arrive z-scored per subject (see ml.sources), so there is one forward
        # signature and it is the one the int8 build is converted from.
        signature = [tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)]
        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.label_shape, dtype=tf.float32),
        ])

        self._init_save_restore()

    def _logits(self, features):
        activation = self.in_layer(features)
        for layer in self.hidden_layers:
            activation = layer(activation)
        return self.out_layer(activation)

    def eval_eager(self, features: tf.Tensor):
        return {'logits': self._logits(features)}

    def train_eager(self, features: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self._logits(features)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class FeatureMLPTrainer(Trainer):
    """Trains the FeatureMLP on the feature vectors the source extracts from each
    subject's mixed signal, against that mix's per-window labels."""

    primary_metric = 'accuracy'
    dataset_tensors = ['features', 'labels']
    n_eval_inputs = 1

    def __init__(self, model: FeatureMLP, data_root: Path):
        super().__init__(model, data_root)
        self.model: FeatureMLP = model # type: ignore

    def subject_arrays(self, sid):
        return (self.data.features(sid, MIXED),
                self.data.window_labels(sid).reshape(-1, 1))

    def calibration_arrays(self):
        return self.calibration.calibration_features()

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt

        tp, fp, tn, fn = 0, 0, 0, 0
        for x, y in eval_dataset:
            pred = tf.cast(self.model.eval(x)['logits'] > 0.0, tf.float32)
            tp += int(tf.reduce_sum(pred * y))
            fp += int(tf.reduce_sum(pred * (1 - y)))
            tn += int(tf.reduce_sum((1 - pred) * (1 - y)))
            fn += int(tf.reduce_sum((1 - pred) * y))

        matrix = [[tn, fp], [fn, tp]]
        labels = ['Normal', 'Anomaly']

        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'Pred {l}' for l in labels])
        ax.set_yticklabels([f'True {l}' for l in labels])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha='center', va='center', fontsize=12)
        fig.colorbar(im)
        fig.tight_layout()
        path = result_dir / 'confusion_matrix.png'
        fig.savefig(path)
        plt.close(fig)
        print(f"saved confusion matrix to {path}")

    def eval_metrics(self, datapoints, outputs):
        correct, total = 0.0, 0.0
        for (_, y), out in zip(datapoints, outputs):
            pred = (np.asarray(out['logits']).reshape(-1) > 0.0)
            y = np.asarray(y).reshape(-1)
            correct += float(np.sum(pred == (y > 0.5)))
            total += float(y.size)
        return {'accuracy': correct / total if total else 0.0}


def get_model(data_root: Path, batch_size: int | None = None) -> FeatureMLP:
    return FeatureMLP(
        name='feature_anomaly',
        batch_size=batch_size or FeatureMLP.default_batch_size,
    )


def get_trainer(data_root: Path, batch_size: int | None = None) -> FeatureMLPTrainer:
    return FeatureMLPTrainer(get_model(data_root, batch_size), data_root)
