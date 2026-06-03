import tensorflow as tf


@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor):
    return tf.reduce_mean((y - x) ** 2)


def fed_avg(vectors: list[tf.Tensor], sizes: list[int]) -> tf.Tensor:
    total = sum(sizes)
    avg = tf.zeros(vectors[0].shape)

    for vector, size in zip(vectors, sizes):
        avg += vector * (size / total)

    return avg
