import math
from typing import Callable

import numpy as np
import tensorflow as tf

def relu_nocustom(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.relu(x)


@tf.custom_gradient
def relu(x: tf.Tensor):
    y = relu_nocustom(x)

    def grad(dy: tf.Tensor):
        return dy * tf.cast(y > 0.0, dy.dtype)

    return y, grad


def upsample2(x: tf.Tensor) -> tf.Tensor:
    _, seq_len, channels = (int(d) for d in x.shape)
    return tf.reshape(tf.stack([x, x], axis=2), [-1, 2 * seq_len, channels])


def conv1d_same_nocustom(x: tf.Tensor, kernel: tf.Tensor, stride: int) -> tf.Tensor:
    return tf.nn.conv1d(x, kernel, stride=stride, padding='SAME')


def _same_padding(seq_len: int, kernel_size: int, stride: int) -> tuple[int, int]:
    out_len = seq_len // stride
    total = max((out_len - 1) * stride + kernel_size - seq_len, 0)
    return total // 2, total - total // 2


def _conv1d_same_grad(dy: tf.Tensor, x: tf.Tensor, kernel: tf.Tensor,
                      stride: int) -> tuple[tf.Tensor, tf.Tensor]:
    batch, seq_len, in_ch = (int(d) for d in x.shape)
    kernel_size, _, out_ch = (int(d) for d in kernel.shape)
    out_len = seq_len // stride
    pad_left, pad_right = _same_padding(seq_len, kernel_size, stride)

    dx = tf.nn.conv1d_transpose(dy, kernel, output_shape=[batch, seq_len, in_ch],
                                strides=stride, padding='SAME')

    x_pad = tf.pad(x, [[0, 0], [pad_left, pad_right], [0, 0]])
    dy_flat = tf.reshape(dy, [batch * out_len, out_ch])
    last_window = (out_len - 1) * stride    # start of the rightmost output window

    # dk[tap, ci, co] = sum over (b, o) of x_pad[b, o * stride + tap, ci] * dy[b, o, co]
    dk_taps = []
    for tap in range(kernel_size):
        # the one input sample per output position that tap multiplies: [batch, out_len, in_ch]
        x_tap = x_pad[:, tap : tap + last_window + 1 : stride, :]
        dk_taps.append(tf.matmul(tf.reshape(x_tap, [batch * out_len, in_ch]),
                                 dy_flat, transpose_a=True))

    return dx, tf.stack(dk_taps)


@tf.custom_gradient
def conv1d_same(x: tf.Tensor, kernel: tf.Tensor, stride: int):
    stride = int(tf.get_static_value(stride))  # type: ignore[arg-type]
    y = conv1d_same_nocustom(x, kernel, stride)

    def grad(dy: tf.Tensor):
        dx, dk = _conv1d_same_grad(dy, x, kernel, stride)
        return dx, dk, None

    return y, grad


class Dense(tf.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: Callable | None =None):
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.weight = tf.Variable(tf.random.uniform(
            shape=[in_dim, out_dim], minval=-limit, maxval=limit
        ), name='dense_weight')
        self.bias = tf.Variable(tf.zeros(shape=[out_dim]), name='dense_bias')
        self.activation = activation if activation else (lambda x: x)

    def __call__(self, data) -> tf.Tensor:
        out = data @ self.weight + self.bias
        return self.activation(out)


class LSTMCell(tf.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim

        limit_w = math.sqrt(6.0 / (in_dim + 4 * hidden_dim))
        self.W = tf.Variable(tf.random.uniform(
            shape=[in_dim, 4 * hidden_dim], minval=-limit_w, maxval=limit_w), name='lstm_W')

        limit_u = math.sqrt(6.0 / (hidden_dim + 4 * hidden_dim))
        self.U = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, 4 * hidden_dim], minval=-limit_u, maxval=limit_u), name='lstm_U')

        self.b = tf.Variable(tf.zeros(shape=[4 * hidden_dim]), name='lstm_b')

    def zero_state(self, batch_size: int):
        h = tf.zeros([batch_size, self.hidden_dim])
        c = tf.zeros([batch_size, self.hidden_dim])
        return h, c

    def step(self, h, c, x_t):
        z = x_t @ self.W + h @ self.U + self.b
        i, f, g, o = tf.split(z, 4, axis=-1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g = tf.tanh(g)
        c_new = f * c + i * g
        h_new = o * tf.tanh(c_new)
        return h_new, c_new


class GRUCell(tf.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim

        limit_w = math.sqrt(6.0 / (in_dim + 3 * hidden_dim))
        self.W = tf.Variable(tf.random.uniform(
            shape=[in_dim, 3 * hidden_dim], minval=-limit_w, maxval=limit_w), name='gru_W')

        limit_zr = math.sqrt(6.0 / (hidden_dim + 2 * hidden_dim))
        self.U_zr = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, 2 * hidden_dim], minval=-limit_zr, maxval=limit_zr), name='gru_U_zr')

        limit_n = math.sqrt(6.0 / (2 * hidden_dim))
        self.U_n = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, hidden_dim], minval=-limit_n, maxval=limit_n), name='gru_U_n')

        self.b = tf.Variable(tf.zeros(shape=[3 * hidden_dim]), name='gru_b')

    def zero_state(self, batch_size: int):
        return tf.zeros([batch_size, self.hidden_dim])

    def step(self, h, x_t):
        xz, xr, xn = tf.split(x_t @ self.W + self.b, 3, axis=-1)
        hz, hr = tf.split(h @ self.U_zr, 2, axis=-1)
        z = tf.sigmoid(xz + hz)
        r = tf.sigmoid(xr + hr)
        n = tf.tanh(xn + (r * h) @ self.U_n)
        return (1.0 - z) * n + z * h


class Conv1D(tf.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 activation: Callable | None = None):
        limit = math.sqrt(6.0 / (kernel_size * (in_ch + out_ch)))
        self.kernel = tf.Variable(
                tf.random.uniform(
                    shape=[kernel_size, in_ch, out_ch],
                    minval=-limit, 
                    maxval=limit
                ), name='conv_kernel')
        self.bias = tf.Variable(tf.zeros(shape=[out_ch]), name='conv_bias')
        self.stride = stride
        self.activation = activation if activation else (lambda x: x)

    def __call__(self, x):
        out = conv1d_same(x, self.kernel, self.stride) + self.bias
        return self.activation(out)


def sinusoidal_encoding(length: int, dim: int) -> tf.Tensor:
    pos = np.arange(length)[:, None]
    idx = np.arange(dim)[None, :]
    angle = pos / np.power(10000.0, (2 * (idx // 2)) / dim)
    pe = np.zeros((length, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)
