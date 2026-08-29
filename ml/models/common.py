from abc import ABC, abstractmethod
from typing import Protocol
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf

from ..optimizers import Adam
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..preprocessing import BVP_RATE, BVP_WINDOW, DESCRIPTOR_PARAMS_FILE
from ..dataset_list import calibration_source, training_source
from ..loading import batched, to_dataset
from ..sources.common import DataSource
from ..sources.dalia import CLEAN
from ..spectral import descriptor


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
    """The model-specific half of a training run: how to shape this model's datapoints,
    how to score them, and how to feed the int8 converter. Everything generic — the
    loops, the splits, the dataset plumbing — lives in ml.training and ml.loading, so any
    model works under any loop.

    A trainer is pinned to the training dataset and offers no way to select another: its
    consumers are the system itself (scripts.system.train, scripts.integration.fed_client,
    worker.tasks), which trains on one dataset by definition. Scripts that already know
    which architecture they hold build a source from ml.dataset_list directly instead."""

    model: TrainableModel
    primary_metric: str
    # Names of the tensors each dataset batch yields, in order — used to match
    # dataset arrays to the model's signature inputs by name (see scripts/fed_client.py).
    dataset_tensors: list[str]
    # How many leading dataset tensors the eval signature consumes; the remaining
    # ones are targets ``eval_metrics`` reads off the datapoints (e.g. the MLP's labels).
    n_eval_inputs: int
    # Fixes how the device feeds the model: the norm_param_bytes layout and the
    # I/O signature semantics. Part of the signed model bytes (see ml.payload).
    contract_version: int

    def __init__(self, model: TrainableModel, data_root: Path):
        self.model = model
        self.data: DataSource = training_source(data_root)
        self.calibration: DataSource = calibration_source(data_root)

    @abstractmethod
    def subject_arrays(self, sid: str) -> tuple[np.ndarray, ...]:
        """One subject's datapoints as raw arrays, one per entry of ``dataset_tensors``."""

    @abstractmethod
    def calibration_arrays(self) -> np.ndarray:
        """Datapoints the int8 converter calibrates its tensor scales on, drawn from the
        unfiltered dataset — the device runs under every activity, not only the ones the
        model trained on."""

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

    def subject_ids(self) -> list[str]:
        return self.data.subject_ids()

    def subject_datasets(self) -> list[tf.data.Dataset]:
        """Every subject's batched dataset, in subject order. Split it with
        ml.loading.holdout and merge it with ml.loading.pool."""
        return [batched(to_dataset(*self.subject_arrays(sid)), self.model.batch_size)
                for sid in self.subject_ids()]

    def representative_dataset(self) -> tf.data.Dataset:
        """Feed-dict stream for the int8 TFLite converter, always built from a small
        sample of the unfiltered dataset on disk — the worker builds one for every model
        at startup and must not window the whole dataset to do it."""
        return (to_dataset(self.calibration_arrays())
                .batch(self.model.batch_size, drop_remainder=True)
                .map(self.normalize_feed))


class TrainerBuilder(Protocol):
    def __call__(self, data_root: Path, batch_size: int | None = None) -> Trainer: ...


class ModelBuilder(Protocol):
    def __call__(self, data_root: Path,
                 batch_size: int | None = None) -> TrainableModel: ...


def autoencoder_norm_params(data_root: Path):
    """z-score params baked into an autoencoder so it normalizes its own raw input:
    the BVP signal. ACC is not a model input — it only feeds feature extraction.
    Computed over every activity: the device normalizes the same way everywhere."""
    return calibration_source(data_root).signal_norm_stats()


def descriptor_norm_params(data_root: Path):
    """z-score params for the spectral descriptor, cached next to the dataset.

    Computed over the same windows the descriptor autoencoder trains on — every
    subject's clean low-activity windows — because the descriptor is the model's input
    space, not the device's: the signal params above already cover what the device feeds
    in. Written on first use and reused afterwards; delete the file to recompute it.
    """
    path = data_root / DESCRIPTOR_PARAMS_FILE
    if path.exists():
        params = np.load(path)
        return params[0], params[1]

    source = training_source(data_root)
    signal_mean, signal_std = source.signal_norm_stats()
    rows = []
    for sid in source.subject_ids():
        windows = source.signal_windows(sid, CLEAN, BVP_WINDOW,
                                        AutoencoderTrainer.default_shift)
        if len(windows):
            rows.append(np.asarray(descriptor(
                tf.constant((windows - signal_mean) / signal_std, tf.float32), BVP_WINDOW)))

    stacked = np.concatenate(rows)
    params = np.stack([stacked.mean(axis=0), stacked.std(axis=0) + 1e-6]).astype(np.float32)
    np.save(path, params)
    print(f"Saved descriptor norm params to {path}")
    return params[0], params[1]


class AutoencoderTrainer(Trainer):

    primary_metric = 'recon_error'
    dataset_tensors = ['signal']
    n_eval_inputs = 1
    contract_version = 2   # norm layout: signal mean/std (1 each)

    default_shift = BVP_RATE * 3 # shift 3 seconds

    def __init__(self, model: TrainableAutoencoder, data_root: Path,
                 shift: int = default_shift):
        super().__init__(model, data_root)
        self.model: TrainableAutoencoder = model # type: ignore
        self.shift = shift

    def norm_param_bytes(self):
        return np.concatenate([
            self.model.signal_mean.numpy(), self.model.signal_std.numpy(),
        ]).astype('<f4').tobytes()

    def subject_arrays(self, sid):
        return (self.data.signal_windows(sid, CLEAN, self.model.seq_len, self.shift),)

    def calibration_arrays(self):
        return self.calibration.calibration_windows(self.model.seq_len, self.shift)

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
