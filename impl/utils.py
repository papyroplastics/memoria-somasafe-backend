import tensorflow as tf

@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor):
    return tf.reduce_mean((y-x)**2)

