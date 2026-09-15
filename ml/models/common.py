from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf

from ..optimizers import Adam
from ..metrics import mse_loss, reconstruction_error
from ..dataset_list import DATASETS
from ..sources.common import DataSource, batched, to_dataset


class UnboundError(NotImplementedError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def unbound(*_, **__):
    raise UnboundError('This function is bound dynamically at init time')


class TrainableModel(tf.Module):
    """Minimal LiteRT-trainable FL contract: the four signatures below and a flat float32
    weight buffer over ``trainable_variables``. ``train`` need only mutate variables and
    return a dict carrying ``loss``."""

    eval: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    train: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    save: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

    default_batch_size: int
    batch_size: int

    def transfer_from(self, source: 'TrainableModel') -> None:
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


class BackpropModel(TrainableModel):
    """Gradient-descent models: owns the Adam and the tape/apply step, so a subclass only
    writes the forward pass and the loss."""

    def _init_optimizer(self, learning_rate: float, beta1: float, beta2: float,
                        epsilon: float):
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

    def _apply_step(self, loss_fn: Callable[[], tf.Tensor]):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class TrainableAutoencoder(BackpropModel):
    """Reconstructs its own input and scores a datapoint by the reconstruction error."""

    default_batch_size = 64

    def __init__(self, name: str, batch_size: int, input_shape: tuple[int, ...]):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.input_shape = (batch_size, *input_shape)

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        """Bind train/eval/save/restore; call once all layers exist."""
        self._init_optimizer(learning_rate, beta1, beta2, epsilon)
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
        return self._apply_step(lambda: self._loss(self._forward(datapoint), target))


class Trainer(ABC):
    """The model-specific half of a training run: how to shape this model's datapoints,
    how to score them, and how to feed the int8 converter."""

    model: TrainableModel
    primary_metric: str
    dataset_tensors: list[str]
    n_eval_inputs: int
    training_key: str
    calibration_key: str | None = None
    shuffle_buffer: int = 1000
    cache_batches: bool = True
    default_holdout: str = 'last:2'

    def __init__(self, model: TrainableModel, data_root: Path):
        self.model = model
        self.data: DataSource = DATASETS[self.training_key].build(data_root)
        self.calibration: DataSource = DATASETS[
            self.calibration_key or self.training_key].build(data_root)

    @abstractmethod
    def subject_arrays(self, sid: str) -> tuple[np.ndarray, ...]:
        """One subject's datapoints, one array per entry of ``dataset_tensors``."""

    def calibration_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.calibration.calibration_data(),)

    @abstractmethod
    def eval_metrics(self, datapoints: list, outputs: list[dict]) -> dict[str, float]:
        """Metrics for this model type from the aligned ``datapoints`` and eval ``outputs``."""

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact."""
        pass

    def arch_fingerprint(self) -> str:
        """Stable hash of the ordered trainable-variable layout."""
        manifest = [
            (var.name, tuple(int(d) for d in var.shape), var.dtype.name)
            for var in self.model.trainable_variables
        ]
        return hashlib.sha256(repr(manifest).encode()).hexdigest()[:16]

    def subject_ids(self) -> list[str]:
        return self.data.subject_ids()

    def subject_datasets(self) -> list[tf.data.Dataset]:
        return [batched(to_dataset(*self.subject_arrays(sid)), self.model.batch_size,
                        self.shuffle_buffer, self.cache_batches)
                for sid in self.subject_ids()]

    def representative_dataset(self) -> tf.data.Dataset:
        names = self.dataset_tensors[:self.n_eval_inputs]
        return (to_dataset(*self.calibration_arrays())
                .batch(self.model.batch_size, drop_remainder=True)
                .map(lambda *tensors: dict(zip(names, tensors))))


class TrainerBuilder(Protocol):
    def __call__(self, data_root: Path, batch_size: int | None = None) -> Trainer: ...


class ModelBuilder(Protocol):
    def __call__(self, data_root: Path,
                 batch_size: int | None = None) -> TrainableModel: ...


class ReconstructionTrainer(Trainer):

    primary_metric = 'recon_error'
    n_eval_inputs = 1

    def subject_arrays(self, sid):
        return (self.data.datapoints(sid),)

    def eval_metrics(self, datapoints, outputs):
        errors = np.concatenate([np.asarray(o['error']).reshape(-1) for o in outputs])
        return {'recon_error': float(np.mean(errors))}
