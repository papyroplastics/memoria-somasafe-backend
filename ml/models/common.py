from abc import ABC, abstractmethod
from typing import Protocol
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf

from ..optimizers import Adam
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..preprocessing import CLEAN_SUBDIR, BVP_RATE
from ..loading import norm_stats, batched, cached, subject_dirs, subject_windows

# Batches of a run's eval set fed to the int8 converter to fix its tensor scales.
CALIBRATION_BATCHES = 150


class UnboundError(NotImplementedError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def unbound(*_, **__):
    raise UnboundError('This function is bound dynamically at init time')


class TrainableModel(tf.Module):
    """Base class for all LiteRT-trainable FL-compatible models.

    Subclasses must:
      1. Create all trainable layers/variables.
      2. Bind ``self.eval`` and ``self.train`` as ``tf.function``s with the
         appropriate ``input_signature``.
      3. Call ``self._init_save_restore()`` once all trainable variables exist
         (optimizer state is non-trainable and need not exist yet).
    """

    eval: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    infer: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    train: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    save: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    default_batch_size: int
    batch_size: int

    def transfer_from(self, source: 'TrainableModel') -> None:
        """Copy ``source``'s trainable variables into this model for transfer
        learning. Both models must share the architecture (same ordered variable
        list) and differ only in things like batch size. Where a variable's shape
        matches it is copied whole; where it differs the overlapping leading
        region is copied and the rest left at this model's init, so a model
        trained at a larger batch size still seeds a smaller one."""
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

    default_batch_size = 64

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 n_outputs, diff_weight, signal_mean, signal_std):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.n_outputs = n_outputs
        self.diff_weight = diff_weight
        self.signal_shape = (batch_size, seq_len, n_signals)

        self.signal_mean = tf.constant(signal_mean, dtype=tf.float32)
        self.signal_std = tf.constant(signal_std, dtype=tf.float32)

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        """Bind train/eval/infer/save/restore. Call once all layers exist."""
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)
        signature = [tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32)]

        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.infer = tf.function(self.infer_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=signature)
        self._init_save_restore()

    def _forward(self, signal: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def _eval_core(self, signal: tf.Tensor):
        """Reconstruction + error from an already-normalized signal."""
        reconstruction = self._forward(signal)
        target = signal[:,:,:self.n_outputs]
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, target)}

    def infer_eager(self, signal: tf.Tensor):
        return self._eval_core(signal)

    def eval_eager(self, signal: tf.Tensor):
        return self._eval_core((signal - self.signal_mean) / self.signal_std)

    def train_eager(self, signal: tf.Tensor):
        signal = (signal - self.signal_mean) / self.signal_std
        target = signal[:,:,:self.n_outputs]
        with tf.GradientTape() as tape:
            reconstruction = self._forward(signal)
            loss = (mse_loss(reconstruction, target)
                    + self.diff_weight * first_difference_loss(reconstruction, target))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class Trainer(ABC):
    """The model-specific half of a training run: how to read this model's data off
    disk, how to score it, and how to feed the int8 converter. Everything generic —
    the loops, the splits, the dataset plumbing — lives in ml.training and ml.loading,
    so any model works under any loop."""

    model: TrainableModel
    primary_metric: str
    data_subdir: str
    # Names of the tensors each dataset batch yields, in order — used to match
    # dataset arrays to the model's signature inputs by name (see scripts/fed_client.py).
    dataset_tensors: list[str]
    # How many leading dataset tensors the eval signature consumes; the remaining
    # ones are targets ``eval_metrics`` reads off the datapoints (e.g. the MLP's labels).
    n_eval_inputs: int
    # Fixes how the device feeds the model: the norm_param_bytes layout and the
    # I/O signature semantics. Part of the signed model bytes (see ml.payload).
    contract_version: int

    @abstractmethod
    def subject_dataset(self, subject_dir: Path) -> tf.data.Dataset:
        """One subject's datapoints, unbatched."""

    @abstractmethod
    def normalize_feed(self, *tensors: tf.Tensor) -> dict[str, tf.Tensor]:
        """One batch as a feed dict for the int8 converter. Calibrates the ``infer``
        graph, which takes already-normalized inputs, so the values must be z-scored
        the way the device feeds them (see saving.optimize_saved_model)."""

    @abstractmethod
    def norm_param_bytes(self) -> bytes:
        """The model's z-score params as LE float32, covered by the server's model
        signature (see ml.payload). Layout is fixed by ``contract_version``; the device
        applies them as ``(x - mean) / std`` before the int8 (non-normalizing) model."""

    @abstractmethod
    def eval_metrics(self, datapoints: list, outputs: list[dict]) -> dict[str, float]:
        """Metrics relevant to this model type (accuracy, recon error, ...) from the
        aligned lists of evaluated ``datapoints`` (each a full dataset batch tuple) and
        per-datapoint eval-signature ``outputs``. Kept independent of the runtime that
        produced the outputs so both the in-process TF path (``ml.training.evaluate``)
        and the on-device LiteRT path (``scripts/integration/fed_client.py``) share it;
        output values may be tf tensors or numpy arrays and target tensors are read off
        ``datapoints``."""

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact."""
        pass

    def dataset_key(self) -> tuple:
        """Everything besides the data root and the subject that changes the datapoints,
        so two trainers only share a cached dataset when it means the same thing."""
        return (type(self).__name__, self.model.batch_size)

    def arch_fingerprint(self) -> str:
        """Stable hash of the weight-compatibility boundary: the ordered
        trainable-variable layout (name/shape/dtype) plus the baked normalization
        params. Two builds share a fingerprint iff their flat parameter buffers
        are semantically interchangeable. Derived from code + data, never
        hand-bumped — the seed script checks the registry version against it."""
        manifest = [
            (var.name, tuple(int(d) for d in var.shape), var.dtype.name)
            for var in self.model.trainable_variables
        ]
        return hashlib.sha256(
            repr(manifest).encode() + self.norm_param_bytes()).hexdigest()[:16]

    def subject_datasets(self, data_root: Path) -> list[tf.data.Dataset]:
        """Every subject's batched, cached dataset, in subject order. Split it with
        ml.loading.holdout and merge it with ml.loading.pool."""
        key = self.dataset_key()
        return [cached((str(data_root), self.data_subdir, d.name, *key),
                       lambda d=d: batched(self.subject_dataset(d), self.model.batch_size))
                for d in subject_dirs(data_root, self.data_subdir)]

    def representative_dataset(self, dataset: tf.data.Dataset | None = None,
                               data_root: Path | None = None) -> tf.data.Dataset:
        """Feed-dict stream for the int8 TFLite converter. Calibrates on ``dataset`` when
        given (a run's eval set), otherwise on a small sample drawn from ``data_root`` —
        the worker builds this for every model at startup and must not window the whole
        dataset to do it."""
        if dataset is None:
            if data_root is None:
                raise ValueError("Either dataset or data_root must be passed")
            dataset = self.calibration_feed(data_root)
        else:
            dataset = dataset.take(CALIBRATION_BATCHES)
        return dataset.map(self.normalize_feed)

    def calibration_feed(self, data_root: Path, per_subject: int = 10) -> tf.data.Dataset:
        """A few random datapoints from each subject, batched — enough to fix the int8
        tensor scales without building the full training pipeline. Sampled across every
        subject rather than off the head of each: all subjects start at rest, so a
        prefix would calibrate on an at-rest range and clip everything above it."""
        parts = []
        for d in subject_dirs(data_root, self.data_subdir):
            ds = self.subject_dataset(d)
            if len(ds):
                parts.append(ds.shuffle(len(ds), reshuffle_each_iteration=False)
                               .take(min(per_subject, len(ds))))
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.concatenate(part)
        return merged.batch(self.model.batch_size, drop_remainder=True)


class TrainerBuilder(Protocol):
    def __call__(self, data_root: Path, batch_size: int | None = None) -> Trainer: ...


def autoencoder_norm_params(data_root: Path, data_subdir: str = CLEAN_SUBDIR):
    """z-score params baked into an autoencoder so it normalizes its own raw input:
    the BVP signal. ACC is not a model input — it only feeds feature extraction."""
    return norm_stats(data_root / data_subdir)

class AutoencoderTrainer(Trainer):

    primary_metric = 'recon_error'
    dataset_tensors = ['signal']
    n_eval_inputs = 1
    contract_version = 2   # norm layout: signal mean/std (1 each)

    default_shift = BVP_RATE * 3

    def __init__(self, model: TrainableAutoencoder, shift: int = default_shift,
                 data_subdir: str = CLEAN_SUBDIR):
        self.model: TrainableAutoencoder = model # type: ignore
        self.shift = shift
        self.data_subdir = data_subdir

    def dataset_key(self):
        return (*super().dataset_key(), self.model.seq_len, self.shift)

    def norm_param_bytes(self):
        return np.concatenate([
            self.model.signal_mean.numpy(), self.model.signal_std.numpy(),
        ]).astype('<f4').tobytes()

    def subject_dataset(self, subject_dir):
        return subject_windows(subject_dir.parent, subject_dir.name,
                               self.model.seq_len, self.shift)

    def normalize_feed(self, signal):
        return {'signal': (signal - self.model.signal_mean) / self.model.signal_std}

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
