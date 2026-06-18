import tensorflow as tf
from typing import Sequence


class MomentumSGD(tf.Module):
    def __init__(self, variables: Sequence[tf.Variable], learning_rate: float, momentum: float):
        super().__init__()
        self.learning_rate = tf.constant(learning_rate)
        self.momentum = tf.constant(momentum)
        self.velocity = [
            tf.Variable(tf.zeros_like(var), trainable=False) for var in variables
        ]

    def apply(self, variables: Sequence[tf.Variable], grads: list):
        for i, var in enumerate(variables):
            self.velocity[i].assign(self.momentum * self.velocity[i] + grads[i])  # type: ignore
            var.assign_sub(self.learning_rate * self.velocity[i])


class Adam(tf.Module):
    def __init__(self, 
                 variables: Sequence[tf.Variable],
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-7):
        super().__init__()
        self.learning_rate = tf.constant(learning_rate)
        self.beta1 = tf.constant(beta1)
        self.beta2 = tf.constant(beta2)
        self.epsilon = tf.constant(epsilon)
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self.step = tf.Variable(tf.constant(0.0), trainable=False)

    def apply(self, variables: Sequence[tf.Variable], grads: list):
        self.step.assign_add(1.0)
        t = self.step
        lr_t = (self.learning_rate
                * tf.sqrt(1.0 - tf.pow(self.beta2, t))
                / (1.0 - tf.pow(self.beta1, t)))
        for i, var in enumerate(variables):
            g = grads[i]
            self.m[i].assign(self.beta1 * self.m[i] + (1.0 - self.beta1) * g)
            self.v[i].assign(self.beta2 * self.v[i] + (1.0 - self.beta2) * tf.square(g))
            var.assign_sub(lr_t * self.m[i] / (tf.sqrt(self.v[i]) + self.epsilon))

