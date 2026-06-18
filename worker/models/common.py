import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from ..optimizers import Adam


@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean((y - x) ** 2)


def reconstruction_error(reconstruction: tf.Tensor, signal: tf.Tensor) -> tf.Tensor:
    """Per-window mean squared error — the anomaly score for autoencoders."""
    return tf.reduce_mean(tf.square(reconstruction - signal), axis=[1, 2])


def window_signal(signal: np.ndarray, window_size: int, shift: int):
    """Window a ``(T, n_signals)`` array into ``(window_size, n_signals)`` frames.

    Returns the windowed dataset and its (asserted) cardinality.
    """
    count = (len(signal) - window_size) // shift + 1
    ds = (tf.data.Dataset.from_tensor_slices(signal)
          .window(size=window_size, shift=shift, drop_remainder=True)
          .flat_map(lambda w: w.batch(window_size, drop_remainder=True))
          .apply(tf.data.experimental.assert_cardinality(count)))
    return ds, count


class UnboundError(NotImplementedError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def unbound(*_, **__):
    raise UnboundError('This function is bound dynamically at init time')


class Dense(tf.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: Callable | None =None):
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.weight = tf.Variable(tf.random.uniform(
            shape=[in_dim, out_dim], minval=-limit, maxval=limit
        ))
        self.bias = tf.Variable(tf.zeros(shape=[out_dim]))
        self.activation = activation if activation else (lambda x: x)

    def __call__(self, data):
        out = data @ self.weight + self.bias
        return self.activation(out)


class LSTMCell(tf.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim

        limit_w = math.sqrt(6.0 / (in_dim + 4 * hidden_dim))
        self.W = tf.Variable(tf.random.uniform(
            shape=[in_dim, 4 * hidden_dim], minval=-limit_w, maxval=limit_w))

        limit_u = math.sqrt(6.0 / (hidden_dim + 4 * hidden_dim))
        self.U = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, 4 * hidden_dim], minval=-limit_u, maxval=limit_u))

        self.b = tf.Variable(tf.zeros(shape=[4 * hidden_dim]))

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
            shape=[in_dim, 3 * hidden_dim], minval=-limit_w, maxval=limit_w))

        limit_zr = math.sqrt(6.0 / (hidden_dim + 2 * hidden_dim))
        self.U_zr = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, 2 * hidden_dim], minval=-limit_zr, maxval=limit_zr))

        limit_n = math.sqrt(6.0 / (2 * hidden_dim))
        self.U_n = tf.Variable(tf.random.uniform(
            shape=[hidden_dim, hidden_dim], minval=-limit_n, maxval=limit_n))

        self.b = tf.Variable(tf.zeros(shape=[3 * hidden_dim]))

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
        self.kernel = tf.Variable(tf.random.uniform(
            shape=[kernel_size, in_ch, out_ch], minval=-limit, maxval=limit))
        self.bias = tf.Variable(tf.zeros(shape=[out_ch]))
        self.stride = stride
        self.activation = activation if activation else (lambda x: x)

    def __call__(self, x):
        out = tf.nn.conv1d(x, self.kernel, stride=self.stride, padding='SAME') + self.bias
        return self.activation(out)


class TrainableModel(tf.Module):
    """Base class for all LiteRT-trainable / FedAvg-compatible models.

    Subclasses must:
      1. Create all trainable layers/variables.
      2. Bind ``self.eval`` and ``self.train`` as ``tf.function``s with the
         appropriate ``input_signature``.
      3. Call ``self._init_save_restore()`` once all trainable variables exist
         (optimizer state is non-trainable and need not exist yet).
    """

    eval: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    train: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    save: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    def _init_save_restore(self):
        self.parameter_sizes = [
            int(var.shape.num_elements()) for var in self.trainable_variables
        ]
        self.total_parameter_size = sum(self.parameter_sizes)
        self.save = tf.function(self.save_eager, input_signature=[])
        self.restore = tf.function(self.restore_eager, input_signature=[
            tf.TensorSpec(shape=(self.total_parameter_size,), dtype=tf.float32),
        ])

    def save_eager(self):
        return {
            'parameters': tf.concat([
                tf.reshape(var, (-1,)) for var in self.trainable_variables
            ], axis=0)
        }

    def restore_eager(self, parameters: tf.Tensor):
        idx = 0
        for i, var in enumerate(self.trainable_variables):
            size = self.parameter_sizes[i]
            var.assign(tf.reshape(parameters[idx:idx + size], var.shape))
            idx += size
        return {
            'parameter_count': tf.constant(self.total_parameter_size, dtype=tf.int32)
        }


class TrainableAutoencoder(TrainableModel):
    """Non-conditional reconstruction autoencoder base.

    Subclasses build their encoder/decoder layers and implement ``_forward``;
    the train/eval bodies, signature binding and Adam optimizer are shared. The
    per-window reconstruction error is the anomaly score. Conditional variants
    (extra demographics/context inputs) stay outside this hierarchy so they can
    be compared against a plain reconstruction model.
    """

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.signal_shape = (batch_size, seq_len, n_signals)

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        """Bind train/eval/save/restore. Call once all layers exist."""
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)
        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32)])
        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32)])
        self._init_save_restore()

    def _forward(self, signal: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def eval_eager(self, signal: tf.Tensor):
        reconstruction = self._forward(signal)
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, signal)}

    def train_eager(self, signal: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = mse_loss(self._forward(signal), signal)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class Trainer(ABC):
    """Adapts a ``TrainableModel`` to the uniform surface the training loops in
    ``training.py`` drive. A trainer owns data preparation, the per-epoch step,
    the metrics relevant to its model type, and the representative dataset for
    int8 export. Loops only ever talk to this interface, so any
    ``(model, trainer)`` pair works with any loop.
    """

    model: TrainableModel
    primary_metric: str

    @abstractmethod
    def subject_datasets(
        self, data_root: Path, seed: int
    ) -> tuple[list[tf.data.Dataset], list[tf.data.Dataset]]:
        """Per-subject ``(train, eval)`` splits — the primitive the federated
        loop consumes directly and the normal loop ``combine``s."""

    @abstractmethod
    def representative_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Feed-dict stream for the int8 TFLite converter."""

    @abstractmethod
    def evaluate(self, dataset: tf.data.Dataset) -> dict[str, float]:
        """Metrics relevant to this model type (accuracy, recon error, ...)."""

    def train_epoch(self, dataset: tf.data.Dataset) -> float:
        """One pass over ``dataset``; returns mean training loss. Datasets yield
        tuples matching ``model.train``'s arguments, so this stays arity-agnostic."""
        total, batches = 0.0, 0
        for batch in dataset:
            total += float(self.model.train(*batch)['loss'])
            batches += 1
        return total / batches if batches else 0.0

    def combine(self, datasets: list[tf.data.Dataset]) -> tf.data.Dataset:
        """Merge per-subject datasets for the non-federated loops. Default is
        uniform sampling; override for weighted/interleaved mixing."""
        return tf.data.Dataset.sample_from_datasets(datasets)

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact (e.g. an AE reconstruction plot)."""
        pass


class AutoencoderTrainer(Trainer):
    """Shared trainer for non-conditional autoencoders (LSTM/GRU/CNN/...).

    Windows the normalized ``[BVP, ACC]`` signals at load time and scores with
    reconstruction error. Conditional variants subclass this and override only
    ``_windowed`` / ``representative_dataset``.
    """

    primary_metric = 'recon_error'

    def __init__(self, model: TrainableModel, window_size: int, shift: int,
                 batch_size: int, train_split: float = 0.975,
                 data_subdir: str = 'normalized-signals'):
        self.model = model
        self.window_size = window_size
        self.shift = shift
        self.batch_size = batch_size
        self.train_split = train_split
        self.data_subdir = data_subdir

    def _subject_signal(self, subject_dir: Path) -> np.ndarray:
        bvp = np.load(subject_dir / 'bvp.npy')
        acc = np.load(subject_dir / 'acc.npy')
        return np.stack([bvp, acc], axis=-1).astype(np.float32)

    def _windowed(self, subject_dir: Path) -> tuple[tf.data.Dataset, int]:
        """One subject's unbatched window tuples. Override to add conditioning."""
        sig_ds, count = window_signal(self._subject_signal(subject_dir),
                                      self.window_size, self.shift)
        return sig_ds.map(lambda s: (s,)), count

    def subject_datasets(self, data_root, seed):
        data_dir = data_root / self.data_subdir
        subject_dirs = sorted(data_dir.glob('S*'))
        if not subject_dirs:
            raise FileNotFoundError(
                f"Signal dataset not found at {data_dir}. Run get-dataset.py first.")

        subj_train, subj_eval = [], []
        for d in subject_dirs:
            ds, count = self._windowed(d)
            ds = ds.shuffle(count, seed=seed).batch(self.batch_size, drop_remainder=True)
            n_train = int(len(ds) * self.train_split)
            subj_train.append(ds.take(n_train))
            subj_eval.append(ds.skip(n_train))
        return subj_train, subj_eval

    def representative_dataset(self, dataset):
        return dataset.take(10).map(lambda s: {'signal': s})

    def evaluate(self, dataset):
        errors = [self.model.eval(*batch)['error'] for batch in dataset]
        return {'recon_error': float(tf.reduce_mean(tf.concat(errors, axis=0)))}

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        for batch in eval_dataset.take(1):
            recon = self.model.eval(*batch)['reconstruction']
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(batch[0][0].numpy())
            axs[0].set_title('Input window [BVP, ACC]')
            axs[1].plot(recon[0].numpy())
            axs[1].set_title('Reconstruction')
            fig.savefig(result_dir / 'reconstruction.png')
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break
