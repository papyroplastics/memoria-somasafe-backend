from pathlib import Path

import numpy as np
import tensorflow as tf

from ..layers import Dense, relu
from .common import TrainableModel, Trainer
from ..data import MIXED_FEATURE_SUBDIR, N_FEATURES, get_sorted_paths, load_feature_stats
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

    def __init__(self, name: str, batch_size: int, feat_mean, feat_std,
                 n_features: int = N_FEATURES,
                 hidden_dim: int = 32, hidden_layers: int = 3, learning_rate: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.in_shape = (batch_size, n_features)
        self.label_shape = (batch_size, 1)

        # Raw features come in; the model z-scores them, so nothing ships or applies
        # normalization params off-model (firmware/app feed raw).
        self.feat_mean = tf.constant(feat_mean, dtype=tf.float32)
        self.feat_std = tf.constant(feat_std, dtype=tf.float32)

        self.in_layer = Dense(n_features, hidden_dim, activation=relu)
        self.out_layer = Dense(hidden_dim, 1)
        self.hidden_layers = [
            Dense(hidden_dim, hidden_dim, activation=relu) for _ in range(hidden_layers)
        ]

        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        signature = [tf.TensorSpec(shape=self.in_shape, dtype=tf.float32)]
        # eval/train z-score raw inputs; infer takes already-normalized inputs and is the
        # only signature exported to the int8 model, so its int8 input calibrates on
        # normalized values (see saving.optimize_saved_model).
        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.infer = tf.function(self.infer_eager, input_signature=signature)
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

    def infer_eager(self, features: tf.Tensor):
        return {'logits': self._logits(features)}

    def eval_eager(self, features: tf.Tensor):
        return {'logits': self._logits((features - self.feat_mean) / self.feat_std)}

    def train_eager(self, features: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self._logits((features - self.feat_mean) / self.feat_std)
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
    dataset_tensors = ['features', 'labels']
    n_eval_inputs = 1
    default_batch_size = 1
    contract_version = 1   # norm layout: mean[17] then std[17], LE float32

    def __init__(self, model: FeatureMLP, batch_size: int = 1):
        self.model: FeatureMLP = model # type: ignore
        self.batch_size = batch_size
        self.data_subdir = MIXED_FEATURE_SUBDIR

    def norm_param_bytes(self):
        return np.concatenate([self.model.feat_mean.numpy(),
                               self.model.feat_std.numpy()]).astype('<f4').tobytes()

    def subject_dataset(self, subject_dir):
        x = np.load(subject_dir / 'features.npy').astype(np.float32)   # raw; model normalizes
        y = np.load(subject_dir / 'labels.npy')

        return tf.data.Dataset.from_tensor_slices((x, y))

    def representative_dataset(self, dataset=None, data_root=None):
        # Calibrates the `infer` graph, which takes already-normalized inputs, so yield
        # z-scored features (matching what the device feeds after normalizing).
        mean = tf.constant(self.model.feat_mean)
        std = tf.constant(self.model.feat_std)
        if dataset is None:
            if data_root is None:
                raise ValueError("Either dataset or data_root must be passed")

            rng = np.random.default_rng()
            data_dir = data_root / self.data_subdir
            all_x, all_y = [], []
            for subject_dir in get_sorted_paths(data_dir):
                x = np.load(subject_dir / 'features.npy').astype(np.float32)
                y = np.load(subject_dir / 'labels.npy')
                idx = rng.choice(len(x), size=min(10, len(x)), replace=False)
                all_x.append(x[idx])
                all_y.append(y[idx])
            dataset = tf.data.Dataset.from_tensor_slices((
                np.concatenate(all_x).astype(np.float32),
                np.concatenate(all_y).astype(np.float32),
            )).batch(self.batch_size, drop_remainder=True)
        else:
            dataset = dataset.take(150)
        return dataset.map(lambda x, y: {'features': (x - mean) / std})

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

def get_trainer(data_root: Path, batch_size: int | None = None) -> FeatureMLPTrainer:
    batch_size = batch_size or FeatureMLPTrainer.default_batch_size

    mean, std = load_feature_stats(data_root / MIXED_FEATURE_SUBDIR)
    model = FeatureMLP(
        name='feature_anomaly',
        batch_size=batch_size,
        feat_mean=mean, feat_std=std,
    )
    return FeatureMLPTrainer(model, batch_size=batch_size)
