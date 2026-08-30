from abc import ABC, abstractmethod
from typing import Protocol
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf

from ..optimizers import Adam
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..preprocessing import BVP_RATE
from ..dataset_list import calibration_source, training_source
from ..loading import batched, to_dataset
from ..sources.common import DataSource
from ..sources.dalia import CLEAN


class UnboundError(NotImplementedError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def unbound(*_, **__):
    raise UnboundError('This function is bound dynamically at init time')


class TrainableModel(tf.Module):
    """Base class for all LiteRT-trainable FL-compatible models. Subclasses create their
    layers, bind ``eval``/``train`` as ``tf.function``s, then call ``_init_save_restore()``.
    Inputs arrive already normalized per subject, so a model holds no normalization constants."""

    eval: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    train: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    save: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    default_batch_size: int
    batch_size: int

    def transfer_from(self, source: 'TrainableModel') -> None:
        """Copy ``source``'s trainable variables into this model for transfer learning, copying only the overlapping region where shapes differ."""
        if len(self.trainable_variables) != len(source.trainable_variables):
            raise ValueError(
                f"variable count mismatch: {len(self.trainable_variables)} vs "
                f"{len(source.trainable_variables)} — models are not the same architecture")

        for dst, src in zip(self.trainable_variables, source.trainable_variables):
            if dst.shape == src.shape:
                dst.assign(src)
                continue
            region = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
            merged = dst.numpy()
            merged[region] = src.numpy()[region]
            dst.assign(merged)

    def _init_save_restore(self):
        self.weight_sizes = [
            int(var.shape.num_elements()) for var in self.trainable_variables
        ]
        self.total_weight_size = sum(self.weight_sizes)
        self.save = tf.function(self.save_eager, input_signature=[])
        self.restore = tf.function(self.restore_eager, input_signature=[
            tf.TensorSpec(shape=(self.total_weight_size,), dtype=tf.float32),
        ])

    def save_eager(self):
        return {
            'weights': tf.concat([
                tf.reshape(var, (-1,)) for var in self.trainable_variables
            ], axis=0)
        }

    def restore_eager(self, weights: tf.Tensor):
        idx = 0
        for i, var in enumerate(self.trainable_variables):
            size = self.weight_sizes[i]
            var.assign(tf.reshape(weights[idx:idx + size], var.shape))
            idx += size

        # signatures must have a return value for conversion
        return { 'placeholder': tf.constant(0, dtype=tf.float32) }


class TrainableAutoencoder(TrainableModel):
    """Reconstructs its own input and scores a datapoint by the reconstruction error.
    Subclasses supply the encoder/decoder in ``_forward`` and, where the target is not
    simply the input, ``_target`` and ``_loss``."""

    default_batch_size = 64

    def __init__(self, name: str, batch_size: int, input_shape: tuple[int, ...]):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.input_shape = (batch_size, *input_shape)

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        """Bind train/eval/save/restore; call once all layers exist."""
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)
        signature = [tf.TensorSpec(shape=self.input_shape, dtype=tf.float32)]

        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=signature)
        self._init_save_restore()

    def _forward(self, datapoint: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _target(self, datapoint: tf.Tensor) -> tf.Tensor:
        return datapoint

    def _loss(self, reconstruction: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
        return mse_loss(reconstruction, target)

    def _eval_core(self, datapoint: tf.Tensor):
        target = self._target(datapoint)
        reconstruction = self._forward(datapoint)
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, target)}

    def _train_core(self, datapoint: tf.Tensor):
        target = self._target(datapoint)
        with tf.GradientTape() as tape:
            loss = self._loss(self._forward(datapoint), target)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class SignalAutoencoder(TrainableAutoencoder):
    """The waveform variants (LSTM/GRU/CNN): reconstruct the BVP window itself, with a
    first-difference (slope) term alongside the MSE that penalizes a flat-line output."""

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int = 1,
                 n_outputs: int = 1, diff_weight: float = 1.0):
        super().__init__(name=name, batch_size=batch_size,
                         input_shape=(seq_len, n_signals))
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.n_outputs = n_outputs
        self.diff_weight = diff_weight

    def _target(self, signal):
        return signal[:, :, :self.n_outputs]

    def _loss(self, reconstruction, target):
        return (mse_loss(reconstruction, target)
                + self.diff_weight * first_difference_loss(reconstruction, target))

    def eval_eager(self, signal: tf.Tensor):
        return self._eval_core(signal)

    def train_eager(self, signal: tf.Tensor):
        return self._train_core(signal)


class DescriptorAutoencoder(TrainableAutoencoder):
    """Autoencoders over a fixed per-window descriptor vector rather than the waveform."""

    def __init__(self, name: str, batch_size: int, n_descriptors: int):
        super().__init__(name=name, batch_size=batch_size, input_shape=(n_descriptors,))
        self.n_descriptors = n_descriptors

    def eval_eager(self, descriptor: tf.Tensor):
        return self._eval_core(descriptor)

    def train_eager(self, descriptor: tf.Tensor):
        return self._train_core(descriptor)


class Trainer(ABC):
    """The model-specific half of a training run: how to shape this model's datapoints,
    how to score them, and how to feed the int8 converter. Generic loop/split/dataset
    plumbing lives in ml.training and ml.loading. Pinned to the training dataset."""

    model: TrainableModel
    primary_metric: str
    # Names of the tensors each dataset batch yields, in order — used to match
    # dataset arrays to the model's signature inputs by name (see scripts/fed_client.py).
    dataset_tensors: list[str]
    # How many leading dataset tensors the eval signature consumes; the remaining
    # ones are targets ``eval_metrics`` reads off the datapoints (e.g. the MLP's labels).
    n_eval_inputs: int

    def __init__(self, model: TrainableModel, data_root: Path):
        self.model = model
        self.data: DataSource = training_source(data_root)
        self.calibration: DataSource = calibration_source(data_root)

    @abstractmethod
    def subject_arrays(self, sid: str) -> tuple[np.ndarray, ...]:
        """One subject's datapoints as raw arrays, one per entry of ``dataset_tensors``."""

    @abstractmethod
    def calibration_arrays(self) -> np.ndarray:
        """Datapoints the int8 converter calibrates its tensor scales on, drawn from the unfiltered dataset."""

    @abstractmethod
    def eval_metrics(self, datapoints: list, outputs: list[dict]) -> dict[str, float]:
        """Metrics relevant to this model type from the aligned lists of evaluated ``datapoints`` and eval-signature ``outputs``."""

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact."""
        pass

    def arch_fingerprint(self) -> str:
        """Stable hash of the ordered trainable-variable layout (name/shape/dtype), shared by two builds iff their weight buffers are interchangeable."""
        manifest = [
            (var.name, tuple(int(d) for d in var.shape), var.dtype.name)
            for var in self.model.trainable_variables
        ]
        return hashlib.sha256(repr(manifest).encode()).hexdigest()[:16]

    def subject_ids(self) -> list[str]:
        return self.data.subject_ids()

    def subject_datasets(self) -> list[tf.data.Dataset]:
        """Every subject's batched dataset, in subject order."""
        return [batched(to_dataset(*self.subject_arrays(sid)), self.model.batch_size)
                for sid in self.subject_ids()]

    def representative_dataset(self) -> tf.data.Dataset:
        """Feed-dict stream for the int8 TFLite converter, built from a small sample of the unfiltered dataset on disk."""
        names = self.dataset_tensors[:self.n_eval_inputs]
        return (to_dataset(self.calibration_arrays())
                .batch(self.model.batch_size, drop_remainder=True)
                .map(lambda *tensors: dict(zip(names, tensors))))


class TrainerBuilder(Protocol):
    def __call__(self, data_root: Path, batch_size: int | None = None) -> Trainer: ...


class ModelBuilder(Protocol):
    def __call__(self, data_root: Path,
                 batch_size: int | None = None) -> TrainableModel: ...


class AutoencoderTrainer(Trainer):

    primary_metric = 'recon_error'
    dataset_tensors = ['signal']
    n_eval_inputs = 1
    default_shift = BVP_RATE * 3 # shift 3 seconds

    def __init__(self, model: SignalAutoencoder, data_root: Path,
                 shift: int = default_shift):
        super().__init__(model, data_root)
        self.model: SignalAutoencoder = model # type: ignore
        self.shift = shift

    def subject_arrays(self, sid):
        return (self.data.signal_windows(sid, CLEAN, self.model.seq_len, self.shift),)

    def calibration_arrays(self):
        return self.calibration.calibration_windows(self.model.seq_len, self.shift)

    def eval_metrics(self, datapoints, outputs):
        errors = np.concatenate([np.asarray(o['error']).reshape(-1) for o in outputs])
        return {'recon_error': float(np.mean(errors))}

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        for batch in eval_dataset.take(1):
            recon = self.model.eval(*batch)['reconstruction']
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(batch[0][0].numpy())
            axs[0].set_title('Input window [BVP]')
            axs[1].plot(recon[0].numpy())
            axs[1].set_title('Reconstruction [BVP]')
            fig.savefig(result_dir / 'reconstruction.png')
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break
