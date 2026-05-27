import tensorflow as tf
from typing import Callable

@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor):
    return tf.reduce_mean((y-x)**2)

def train_loop(
        model: Callable[[tf.Tensor], dict],
        train_f: Callable[[tf.Tensor, tf.Tensor], dict],
        train_dataset: tf.data.Dataset,
        eval_dataset: tf.data.Dataset,
        epochs: int):

    for epoch in range(epochs):
        train_loss = 0.0

        for batch_x, batch_y  in train_dataset:
            train_loss = train_f(batch_x, batch_y)['loss']

        if epoch % 10 == 0:
            eval_loss = tf.reduce_mean([mse_loss(model(vb_x)['result'], vb_y) for vb_x, vb_y in eval_dataset], 0)

            print(f"epoch={epoch:03d} train loss={train_loss:.6f} eval_loss={eval_loss:.6f}")

