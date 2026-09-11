import numpy as np
import pytest
import tensorflow as tf

from ..layers import (Conv1D, conv1d_same, conv1d_same_nocustom, relu,
                      relu_nocustom)
from ..models.cnn_autoencoder import CNNAutoencoder

# CNNAutoencoder's convolutions, in encoder-then-decoder order, scaled down by 8 in
# sequence length: (seq_len, in_ch, out_ch, stride). hidden_dim=8 stands in for 32.
KERNEL_SIZE = 5
CONV_CASES = [
    (64, 1, 8, 1),   # enc_in
    (64, 8, 8, 1),   # enc block, stride 1
    (64, 8, 8, 2),   # enc block, downsampling
    (32, 8, 8, 1),
    (32, 8, 8, 2),
    (16, 8, 8, 2),
    (8, 8, 8, 2),
    (64, 8, 1, 1),   # dec_out
]
BATCH = 2

TOL = dict(rtol=2e-4, atol=1e-5)


def vjp(fn, inputs: list[tf.Tensor], dy: tf.Tensor) -> list[np.ndarray]:
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        y = fn(*inputs)
        loss = tf.reduce_sum(y * dy)
    return [g.numpy() for g in tape.gradient(loss, inputs)]


def conv_inputs(seq_len: int, in_ch: int, out_ch: int, stride: int):
    x = tf.random.normal([BATCH, seq_len, in_ch])
    kernel = tf.random.normal([KERNEL_SIZE, in_ch, out_ch])
    dy = tf.random.normal([BATCH, seq_len // stride, out_ch])
    return x, kernel, dy


@pytest.mark.parametrize('shape', [(BATCH, 64, 8), (BATCH, 16), (BATCH, 8, 1)])
def test_relu_matches_builtin(shape):
    x = tf.random.normal(shape)
    dy = tf.random.normal(shape)

    (custom,) = vjp(relu, [x], dy)
    (builtin,) = vjp(relu_nocustom, [x], dy)

    np.testing.assert_allclose(custom, builtin, **TOL)


def test_relu_gradient_at_zero():
    x = tf.constant([[-1.0, 0.0, 1.0]])
    dy = tf.ones_like(x)

    (custom,) = vjp(relu, [x], dy)
    (builtin,) = vjp(relu_nocustom, [x], dy)

    np.testing.assert_array_equal(custom, builtin)
    np.testing.assert_array_equal(custom, [[0.0, 0.0, 1.0]])


@pytest.mark.parametrize('case', CONV_CASES, ids=lambda c: 'len{}_{}to{}_s{}'.format(*c))
def test_conv1d_same_forward_matches_builtin(case):
    x, kernel, _ = conv_inputs(*case)
    np.testing.assert_allclose(conv1d_same(x, kernel, case[3]).numpy(),
                               conv1d_same_nocustom(x, kernel, case[3]).numpy(), **TOL)


@pytest.mark.parametrize('case', CONV_CASES, ids=lambda c: 'len{}_{}to{}_s{}'.format(*c))
def test_conv1d_same_gradient_matches_builtin(case):
    stride = case[3]
    x, kernel, dy = conv_inputs(*case)

    dx, dk = vjp(lambda a, k: conv1d_same(a, k, stride), [x, kernel], dy)
    dx_ref, dk_ref = vjp(lambda a, k: conv1d_same_nocustom(a, k, stride),
                         [x, kernel], dy)

    np.testing.assert_allclose(dx, dx_ref, **TOL)
    np.testing.assert_allclose(dk, dk_ref, **TOL)


@pytest.mark.parametrize('case', CONV_CASES, ids=lambda c: 'len{}_{}to{}_s{}'.format(*c))
def test_conv1d_same_gradient_matches_builtin_in_graph_mode(case):
    stride = case[3]
    x, kernel, dy = conv_inputs(*case)

    @tf.function
    def grads(x, kernel, dy):
        with tf.GradientTape() as tape:
            tape.watch([x, kernel])
            loss = tf.reduce_sum(conv1d_same(x, kernel, stride) * dy)
        return tape.gradient(loss, [x, kernel])

    dx, dk = (g.numpy() for g in grads(x, kernel, dy))
    dx_ref, dk_ref = vjp(lambda a, k: conv1d_same_nocustom(a, k, stride),
                         [x, kernel], dy)

    np.testing.assert_allclose(dx, dx_ref, **TOL)
    np.testing.assert_allclose(dk, dk_ref, **TOL)


@pytest.mark.parametrize('stride', [1, 2])
def test_conv1d_layer_gradient_matches_builtin(stride):
    layer = Conv1D(8, 8, KERNEL_SIZE, stride=stride, activation=relu)
    x = tf.random.normal([BATCH, 32, 8])
    dy = tf.random.normal([BATCH, 32 // stride, 8])
    variables = [layer.kernel, layer.bias]

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(x) * dy)
    custom = [g.numpy() for g in tape.gradient(loss, variables)]

    with tf.GradientTape() as tape:
        out = relu_nocustom(conv1d_same_nocustom(x, layer.kernel, stride) + layer.bias)
        loss = tf.reduce_sum(out * dy)
    builtin = [g.numpy() for g in tape.gradient(loss, variables)]

    for grad, ref in zip(custom, builtin):
        np.testing.assert_allclose(grad, ref, **TOL)


def test_cnn_autoencoder_gradients_match_builtin(monkeypatch):
    model = CNNAutoencoder(name='test_cnn_ae', batch_size=BATCH, seq_len=64,
                           hidden_dim=8, latent_dim=6, kernel_size=KERNEL_SIZE)
    signal = tf.random.normal(model.input_shape)
    target = model._target(signal)

    def model_grads():
        with tf.GradientTape() as tape:
            loss = model._loss(model._forward(signal), target)
        return loss, [g.numpy() for g in tape.gradient(loss, model.trainable_variables)]

    loss, custom = model_grads()

    # swap both custom-gradient ops for their builtin equivalents: ``conv1d_same`` is
    # looked up on the layers module by ``Conv1D.__call__``, while ``relu`` was
    # captured as each layer's ``activation`` at construction time.
    from .. import layers
    monkeypatch.setattr(layers, 'conv1d_same', conv1d_same_nocustom)
    for module in model.submodules:
        if getattr(module, 'activation', None) is relu:
            module.activation = relu_nocustom

    loss_ref, builtin = model_grads()

    np.testing.assert_allclose(loss.numpy(), loss_ref.numpy(), **TOL)
    assert len(custom) == len(builtin) > 0
    for var, grad, ref in zip(model.trainable_variables, custom, builtin):
        scale = max(float(np.abs(ref).max()), 1e-6)
        np.testing.assert_allclose(grad / scale, ref / scale, rtol=2e-4, atol=1e-5,
                                   err_msg=f'gradient mismatch for {var.name}')
