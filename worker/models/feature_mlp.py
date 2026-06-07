from pathlib import Path

import numpy as np
import tensorflow as tf

from .common import Dense, TrainableModel
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
        self.out_layer = Dense(hidden_dim, 1, activation=None)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32, name="feature_0")
        ])

        self._init_save_restore()

        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.in_shape, dtype=tf.float32, name="feature_0"),
            tf.TensorSpec(shape=self.label_shape, dtype=tf.float32, name="label_0"),
        ])

    def _logits(self, data):
        activation = self.in_layer(data)
        for layer in self.hidden_layers:
            activation = layer(activation)
        return self.out_layer(activation)

    def eval_eager(self, data: tf.Tensor):
        logits = self._logits(data)
        return {'logit_0': logits, 'score_0': tf.sigmoid(logits)}

    def train_eager(self, data: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self._logits(data)
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss_0': loss}


def _accuracy(model, dataset: tf.data.Dataset) -> float:
    correct, total = 0.0, 0.0
    for x, y in dataset:
        pred = tf.cast(model.eval(x)['score'] > 0.5, tf.float32)
        correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32)))
        total += float(y.shape[0])
    return correct / total if total else 0.0


def classifier_train_loop(model, train_dataset, eval_dataset, epochs):
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


def run(result_dir: Path):
    import matplotlib.pyplot as plt
    from ..saving import save_odt, save_opti

    tf.random.set_seed(1234)

    feature_dir = Path('datasets') / 'ppg-dalia-processed' / 'feature-anomaly'
    feature_file = feature_dir / 'features.npy'
    label_file = feature_dir / 'labels.npy'
    if not feature_file.exists() or not label_file.exists():
        raise SystemExit(f"Feature dataset not found at {feature_dir}. Run get-dataset.py first.")

    batch_size = 64
    train_split = 0.8

    x = np.load(feature_file)
    y = np.load(label_file)

    dataset = (tf.data.Dataset.from_tensor_slices((x, y))
               .shuffle(len(x), seed=1234).batch(batch_size, drop_remainder=True))
    train_count = int(len(dataset) * train_split)
    train_dataset = dataset.take(train_count)
    eval_dataset = dataset.skip(train_count)

    model = FeatureMLP(
        name='feature_anomaly', batch_size=batch_size, n_features=x.shape[1],
        hidden_dim=32, hidden_layers=1, learning_rate=1e-3,
    )

    saved_model, sm_path = save_odt(result_dir, 'pre-train', model)
    print(f"Saved untrained model to {sm_path}")
    rep_dataset = eval_dataset.map(lambda d, l: {'data': d})
    save_opti(result_dir, 'pre-train', model, rep_dataset)

    history = classifier_train_loop(model, train_dataset, eval_dataset, epochs=50)

    saved_model, sm_path = save_odt(result_dir, 'post-train', model)
    print(f"Saved trained model to {sm_path}")
    rep_dataset = eval_dataset.map(lambda d, l: {'data': d})
    save_opti(result_dir, 'post-train', model, rep_dataset)

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
