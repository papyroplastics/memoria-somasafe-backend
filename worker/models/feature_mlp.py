from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .common import Dense, TrainableModel
from ..optimizers import Adam
from ..saving import save_tainable_model, save_optimized_model


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
        self.out_layer = Dense(hidden_dim, 1, activation=None)

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
        logits = self._logits(features)
        return {'logit': logits, 'score': tf.sigmoid(logits)}

    def train_eager(self, features: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self._logits(features)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


def load_feature_dataset(data_root: Path, batch_size: int, seed: int, train_split: float = 0.8):
    feature_dir = data_root / 'feature-anomaly'
    feature_file = feature_dir / 'features.npy'
    label_file = feature_dir / 'labels.npy'
    if not feature_file.exists() or not label_file.exists():
        raise FileNotFoundError(f"Feature dataset not found at {feature_dir}. Run get-dataset.py first.")

    x = np.load(feature_file)
    y = np.load(label_file)

    dataset = (tf.data.Dataset.from_tensor_slices((x, y))
               .shuffle(len(x), seed=seed).batch(batch_size, drop_remainder=True))

    train_count = int(len(dataset) * train_split)

    return dataset.take(train_count), dataset.skip(train_count)

def get_rep_dataset_feed(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.take(10).map(lambda d, l: {'features': d})

def _accuracy(model, dataset: tf.data.Dataset) -> float:
    correct, total = 0.0, 0.0
    for x, y in dataset:
        pred = tf.cast(model.eval(x)['score'] > 0.5, tf.float32)
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
        total += float(y.shape[0])
    return correct / total if total else 0.0


def train_loop(model, train_dataset, eval_dataset, epochs):
    history = []
    for epoch in range(epochs):
        loss = 0.0
        for x, y in train_dataset:
            loss = model.train(x, y)['loss']
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            acc = _accuracy(model, eval_dataset)
            history.append((epoch, float(loss), acc))
            print(f"epoch={epoch:03d} loss={loss:.4f} eval_acc={acc:.3f}")
    return history

def run(data_root: Path, result_dir: Path, seed: int):
    tf.random.set_seed(seed)

    batch_size = 1
    epochs = 15

    train_dataset, eval_dataset = load_feature_dataset(data_root, batch_size, seed)
    rep_dataset = get_rep_dataset_feed(eval_dataset)

    n_features = next(iter(eval_dataset))[0].shape[-1]

    model = FeatureMLP(
        name='feature_anomaly',
        batch_size=batch_size,
        n_features=n_features,
        hidden_dim=32,
        hidden_layers=3,
        learning_rate=1e-3,
    )

    saved_model, sm_path = save_tainable_model(result_dir, 'pre-train', model)
    save_optimized_model(result_dir, 'pre-train', model, rep_dataset)
    print(f"Saved untrained model to {sm_path}")

    history = train_loop(model, train_dataset, eval_dataset, epochs)

    saved_model, sm_path = save_tainable_model(result_dir, 'post-train', model)
    save_optimized_model(result_dir, 'post-train', model, rep_dataset)
    print(f"Saved trained model to {sm_path}")

    epochs, losses, accs = zip(*history)
    fig, ax = plt.subplots()
    ax.plot(epochs, losses, 'b-', label='train loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss', color='b')
    ax2 = ax.twinx()
    ax2.plot(epochs, accs, 'g-', label='eval accuracy')
    ax2.set_ylabel('eval accuracy', color='g')
    ax.set_title('FeatureMLP anomaly classifier')
    fig.savefig(result_dir / 'training.png')
    print(f"saved training plot to {result_dir / 'training.png'}")
