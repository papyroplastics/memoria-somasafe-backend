import math
from pathlib import Path

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


def basic_train_loop(model, train_dataset: tf.data.Dataset,
                     eval_dataset: tf.data.Dataset, epochs: int):
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_x, batch_y in train_dataset:
            train_loss = model.train(batch_x, batch_y)['loss']

        if epoch % 10 == 0:
            eval_loss = tf.reduce_mean(
                [mse_loss(model.eval(vx)['result'], vy) for vx, vy in eval_dataset], 0)
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} eval_loss={eval_loss:.6f}")


def run(data_root: Path, result_dir: Path, seed: int):
    """``sin(x)`` reference demo: trains, exports, fine-tunes the loaded model,
    and verifies flatten-based weight transfer (the FedAvg primitive)."""
    import matplotlib.pyplot as plt
    from ..saving import save_tainable_model, save_optimized_model

    tf.random.set_seed(seed)

    dataset_size, dataset_split, batch_size = 500, 0.8, 50

    train_x = tf.random.uniform([dataset_size, 1], 0, 2.0 * math.pi, tf.float32)
    true_y = tf.sin(train_x)

    dataset = (tf.data.Dataset.from_tensor_slices((train_x, true_y))
               .shuffle(dataset_size).batch(batch_size, drop_remainder=True))
    train_dataset = dataset.take(int(len(dataset) * dataset_split))
    eval_dataset = dataset.skip(int(len(dataset) * dataset_split))

    model = MLP(
        name='sine', batch_size=batch_size,
        in_dim=1, out_dim=1, hidden_dim=64, hidden_layers=2,
        learning_rate=0.01, momentum=0.9,
    )

    basic_train_loop(model, train_dataset, eval_dataset, epochs=10)

    saved_model, _ = save_tainable_model(result_dir, 'pre-train', model)
    rep_dataset = eval_dataset.map(lambda x, y: {'data': x})
    save_optimized_model(result_dir, 'pre-train', model, rep_dataset)

    # Fine-tune the reloaded SavedModel, then transfer its weights back via
    # the flatten save/restore path to confirm they round-trip exactly.
    basic_train_loop(saved_model, train_dataset, eval_dataset, epochs=100)

    test_x = tf.reshape(tf.range(0, 1, delta=1 / 100, dtype=tf.float32) * 2 * math.pi, (-1, 1))
    test_dataset = tf.data.Dataset.from_tensor_slices(test_x).batch(batch_size, drop_remainder=True)
    pred_saved = tf.concat([saved_model.eval(bx)['result'] for bx in test_dataset], 0)

    model.restore(saved_model.save()['parameters'])
    pred_restored = tf.concat([model.eval(bx)['result'] for bx in test_dataset], 0)
    print(f"restore max abs error={tf.reduce_max(tf.abs(pred_restored - pred_saved)):.8f}")

    save_tainable_model(result_dir, 'post-train', model)
    save_optimized_model(result_dir, 'post-train', model, rep_dataset)

    used_x = test_x[:len(pred_saved)]
    plt.figure()
    plt.plot(used_x, tf.sin(used_x), 'r-', label='True function')
    plt.plot(used_x, pred_saved, 'm-', label='Fine-tuned saved model')
    plt.plot(used_x, pred_restored, 'y--', label='Restored model')
    plt.legend()
    plt.savefig(result_dir / 'sine.png')
    print(f"saved plot to {result_dir / 'sine.png'}")
