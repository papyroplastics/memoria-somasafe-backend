import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..layers import Dense
from .common import TrainableModel, Trainer
from ..data import MIXED_FEATURE_SUBDIR, N_FEATURES, get_sorted_paths
from ..optimizers import Adam


class FeatureMLP(TrainableModel):
    """Supervised binary anomaly classifier over hand-crafted window features.

    Option A of the lightweight roadmap: a small Dense-only network mapping an
    on-device-cheap feature vector to a single anomaly logit. Dense-only so it
    stays fully int8-quantizable for TFLM on the ESP32, and trained on
    synthetic anomalies so it yields a clean accuracy curve. Keeps the same
    eval/train/save/restore signatures as the other models for LiteRT training
    and FedAvg weight transfer.
    """

    def __init__(self, name: str, batch_size: int, n_features: int = N_FEATURES,
                 hidden_dim: int = 32, hidden_layers: int = 3, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.in_shape = (batch_size, n_features)
        self.label_shape = (batch_size, 1)

        self.in_layer = Dense(n_features, hidden_dim, activation=tf.nn.relu)
        self.out_layer = Dense(hidden_dim, 1)
        self.hidden_layers = [
            Dense(hidden_dim, hidden_dim, activation=tf.nn.relu) for _ in range(hidden_layers)
        ]

        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)
        ])
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
    """Trains the FeatureMLP on the per-subject feature dataset. Both features and
    labels are read from ``<data_root>/mixed-features/S*/``; to train against
    distilled teacher labels instead of the synthetic ground truth, point the data
    root at a directory with the same structure (see ``distill_labels.py``)."""

    primary_metric = 'accuracy'
    default_batch_size = 1

    def __init__(self, model: FeatureMLP, batch_size: int = 1,
                 train_split: float = 0.8):
        self.model = model
        self.batch_size = batch_size
        self.train_split = train_split
        self.data_subdir = MIXED_FEATURE_SUBDIR

    def subject_dataset(self, subject_dir):
        x = np.load(subject_dir / 'features.npy')
        y = np.load(subject_dir / 'labels.npy')

        return tf.data.Dataset.from_tensor_slices((x, y))

    def representative_dataset(self, dataset=None, *, data_root=None):
        if dataset is None:
            rng = np.random.default_rng()
            data_dir = data_root / self.data_subdir
            all_x, all_y = [], []
            for subject_dir in get_sorted_paths(data_dir):
                x = np.load(subject_dir / 'features.npy')
                y = np.load(subject_dir / 'labels.npy')
                idx = rng.choice(len(x), size=min(10, len(x)), replace=False)
                all_x.append(x[idx])
                all_y.append(y[idx])
            dataset = tf.data.Dataset.from_tensor_slices((
                np.concatenate(all_x).astype(np.float32),
                np.concatenate(all_y).astype(np.float32),
            ))
        else:
            dataset = dataset.take(150)
        return dataset.map(lambda x, y: {'features': x})

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
                ax.text(j, i, matrix[i][j], ha='center', va='center', fontsize=12)
        fig.colorbar(im)
        fig.tight_layout()
        path = result_dir / 'confusion_matrix.png'
        fig.savefig(path)
        plt.close(fig)
        print(f"saved confusion matrix to {path}")

    def evaluate(self, dataset, prefix=''):
        correct, total = 0.0, 0.0
        for x, y in tqdm(dataset, total=len(dataset),
                         desc=f'{prefix} eval'.strip(), leave=False):
            pred = tf.cast(self.model.eval(x)['logits'] > 0.0, tf.float32)
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
            total += float(y.shape[0])
        return {'accuracy': correct / total if total else 0.0}


def get_trainer(batch_size: int | None = None) -> FeatureMLPTrainer:
    batch_size = batch_size or FeatureMLPTrainer.default_batch_size

    model = FeatureMLP(
        name='feature_anomaly',
        batch_size=batch_size,
    )
    return FeatureMLPTrainer(model, batch_size=batch_size)
