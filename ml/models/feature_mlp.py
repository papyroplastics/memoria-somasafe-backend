from pathlib import Path
import numpy as np
import tensorflow as tf

from .common import Dense, TrainableModel, Trainer
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

    def __init__(self, name: str, batch_size: int, n_features: int,
                 hidden_dim: int, hidden_layers: int, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.in_shape = (batch_size, n_features)
        self.label_shape = (batch_size, 1)

        self.in_layer = Dense(n_features, hidden_dim, activation=tf.nn.relu)
        self.hidden_layers = [
            Dense(hidden_dim, hidden_dim, activation=tf.nn.relu) for _ in range(hidden_layers)
        ]
        self.out_layer = Dense(hidden_dim, 1)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)
        ])

        self._init_save_restore()

        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.label_shape, dtype=tf.float32),
        ])

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
    """Trains the FeatureMLP on the per-subject feature dataset. ``label_dir``
    overrides where labels are read from (used by the distillation script to
    feed teacher pseudo-labels instead of the synthetic ground truth);
    features always come from the feature dataset."""

    primary_metric = 'accuracy'

    def __init__(self, model: FeatureMLP, batch_size: int = 1,
                 train_split: float = 0.8, label_dir: str | None = None):
        self.model = model
        self.batch_size = batch_size
        self.train_split = train_split
        self.label_dir = label_dir

    def subject_datasets(self, data_root, seed):
        feature_dir = data_root / 'feature-anomaly'
        subject_dirs = sorted(feature_dir.glob('S*'))
        if not subject_dirs:
            raise FileNotFoundError(
                f"Feature dataset not found at {feature_dir}. Run get-dataset.py first.")

        subj_train, subj_eval = [], []
        for d in subject_dirs:
            x = np.load(d / 'features.npy')
            if self.label_dir is not None:
                y = np.load(data_root / self.label_dir / d.name / 'labels.npy')
            else:
                y = np.load(d / 'labels.npy')

            ds = (tf.data.Dataset.from_tensor_slices((x, y))
                  .shuffle(len(x), seed=seed)
                  .batch(self.batch_size, drop_remainder=True))
            n_train = int(len(ds) * self.train_split)
            subj_train.append(ds.take(n_train))
            subj_eval.append(ds.skip(n_train))
        return subj_train, subj_eval

    def representative_dataset(self, dataset):
        return dataset.take(10).map(lambda x, y: {'features': x})

    def evaluate(self, dataset):
        correct, total = 0.0, 0.0
        for x, y in dataset:
            pred = tf.cast(self.model.eval(x)['logits'] > 0.0, tf.float32)
            correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
            total += float(y.shape[0])
        return {'accuracy': correct / total if total else 0.0}


def get_trainer(data_root: Path, seed: int, label_dir: str | None = None) -> FeatureMLPTrainer:
    feature_dir = data_root / 'feature-anomaly'
    subject_dirs = sorted(feature_dir.glob('S*'))
    if not subject_dirs:
        raise FileNotFoundError(
            f"Feature dataset not found at {feature_dir}. Run get-dataset.py first.")

    n_features = int(np.load(subject_dirs[0] / 'features.npy').shape[-1])

    model = FeatureMLP(
        name='feature_anomaly',
        batch_size=1,
        n_features=n_features,
        hidden_dim=32,
        hidden_layers=3,
        learning_rate=1e-3,
    )
    return FeatureMLPTrainer(model, batch_size=1, label_dir=label_dir)
