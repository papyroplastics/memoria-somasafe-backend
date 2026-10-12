from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf

from common.config import CALIBRATION_DIR
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


def arch_fingerprint(model: 'TrainableModel') -> str:
    manifest = [
        (var.name, tuple(int(d) for d in var.shape), var.dtype.name)
        for var in model.trainable_variables
    ]
    return hashlib.sha256(repr(manifest).encode()).hexdigest()[:16]


class TrainableModel(tf.Module):
    """LiteRT-trainable FL contract: eval/train/save/restore over a flat float32 buffer."""

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

    def reset_optimizer(self) -> None:
        pass

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

    def _init_optimizer(self, learning_rate: float, beta1: float, beta2: float,
                        epsilon: float):
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)

    def reset_optimizer(self) -> None:
        self.optimizer.reset()

    def _apply_step(self, loss_fn: Callable[[], tf.Tensor]):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class TrainableAutoencoder(BackpropModel):

    default_batch_size = 64

    def __init__(self, name: str, batch_size: int | None, input_shape: tuple[int, ...]):
        super().__init__(name=name)
        self.batch_size = batch_size or self.default_batch_size
        self.input_shape = (self.batch_size, *input_shape)

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
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

    model: TrainableModel
    primary_metric: str
    dataset_tensors: list[str]
    n_eval_inputs: int
    training_key: str
    calibration_key: str | None = None
    eval_key: str | None = None
    shuffle_buffer: int = 1000
    cache_batches: bool = True
    default_holdout: str = 'last:2'

    def __init__(self, model: TrainableModel, data_root: Path, training_key: str | None = None):
        self.model = model
        self.data_root = data_root
        if training_key is not None:
            self.training_key = training_key
        self.data: DataSource = DATASETS[self.training_key].build(data_root)

    @abstractmethod
    def subject_arrays(self, sid: str, source: DataSource | None = None) -> tuple[np.ndarray, ...]:
        """One subject's datapoints from ``source`` (default the training source)."""

    def calibration_arrays(self) -> tuple[np.ndarray, ...]:
        key = self.calibration_key or self.training_key
        path = CALIBRATION_DIR / f'{key}.npy'
        if not path.exists():
            raise FileNotFoundError(
                f"no calibration artifact for '{key}' at {path}; run "
                f"scripts/system/get_dataset.py to generate it")
        return (np.load(path),)

    @abstractmethod
    def eval_metrics(self, datapoints: list, outputs: list[dict]) -> dict[str, float]:
        """Metrics for this model type from the aligned ``datapoints`` and eval ``outputs``."""

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        pass

    def subject_ids(self) -> list[str]:
        return self.data.subject_ids()

    def subject_datasets(self, source: DataSource | None = None) -> list[tf.data.Dataset]:
        source = source or self.data
        return [batched(to_dataset(*self.subject_arrays(sid, source)), self.model.batch_size,
                        self.shuffle_buffer, self.cache_batches)
                for sid in source.subject_ids()]

    def eval_datasets(self) -> list[tf.data.Dataset]:
        if self.eval_key is None:
            return []
        return self.subject_datasets(DATASETS[self.eval_key].build(self.data_root))

    def representative_dataset(self) -> tf.data.Dataset:
        names = self.dataset_tensors[:self.n_eval_inputs]
        return (to_dataset(*self.calibration_arrays())
                .batch(self.model.batch_size, drop_remainder=True)
                .map(lambda *tensors: dict(zip(names, tensors))))

class ReconstructionTrainer(Trainer):

    primary_metric = 'recon_error'
    n_eval_inputs = 1

    def subject_arrays(self, sid, source=None):
        return ((source or self.data).datapoints(sid),)

    def eval_metrics(self, datapoints, outputs):
        errors = np.concatenate([np.asarray(o['error']).reshape(-1) for o in outputs])
        return {'recon_error': float(np.mean(errors))}


class ClassifierTrainer(Trainer):

    primary_metric = 'accuracy'
    dataset_tensors = ['features', 'labels']
    n_eval_inputs = 1
    default_holdout = 'none'

    def subject_arrays(self, sid, source=None):
        source = source or self.data
        labels = source.labels(sid)
        assert labels is not None
        return (source.datapoints(sid), labels)

    def eval_metrics(self, datapoints, outputs):
        correct, total = 0.0, 0.0
        for (_, y), out in zip(datapoints, outputs):
            pred = np.argmax(np.asarray(out['logits']), axis=-1)
            truth = np.argmax(np.asarray(y), axis=-1)
            correct += float(np.sum(pred == truth))
            total += float(len(truth))
        return {'accuracy': correct / total if total else 0.0}
