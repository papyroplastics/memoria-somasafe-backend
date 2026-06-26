import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from ..optimizers import Adam
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..data import (DatasetUnavailibleError, SUBJECTS_SUBDIR,
                    windowed_normalized)


class UnboundError(NotImplementedError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def unbound(*_, **__):
    raise UnboundError('This function is bound dynamically at init time')


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

    def arch_fingerprint(self) -> str:
        """Stable hash of the trainable-variable layout (ordered name/shape/dtype).
        Two models share a fingerprint iff their flat parameter buffers are
        interchangeable, so it is the weight-compatibility boundary. Not a
        ``tf.function`` — it inspects the Python-side variable list directly."""
        manifest = [
            (var.name, tuple(int(d) for d in var.shape), var.dtype.name)
            for var in self.trainable_variables
        ]
        return hashlib.sha256(repr(manifest).encode()).hexdigest()[:16]

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
    encoder sees ``n_signals`` channels (``[BVP, ACC]``) but the decoder only
    reconstructs the first ``n_outputs`` (BVP) — ACC is exogenous context that
    explains motion artifacts and is not part of the anomaly score. The objective
    is reconstruction MSE plus a first-difference term that penalizes flat output.
    Conditional variants stay outside this hierarchy so they can be compared
    against a plain reconstruction model.
    """

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 n_outputs: int = 1, diff_weight: float = 1.0):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.n_outputs = n_outputs
        self.diff_weight = diff_weight
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
        target = signal[..., :self.n_outputs]
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, target)}

    def train_eager(self, signal: tf.Tensor):
        target = signal[..., :self.n_outputs]
        with tf.GradientTape() as tape:
            reconstruction = self._forward(signal)
            loss = (mse_loss(reconstruction, target)
                    + self.diff_weight * first_difference_loss(reconstruction, target))
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
    default_batch_size: int
    batch_size: int

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
    def evaluate(self, dataset: tf.data.Dataset, prefix: str = '') -> dict[str, float]:
        """Metrics relevant to this model type (accuracy, recon error, ...).
        ``prefix`` labels the progress bar (e.g. ``epoch=3/20``)."""

    def train_epoch(self, dataset: tf.data.Dataset, prefix: str = '') -> float:
        """One pass over ``dataset``; returns mean training loss. Datasets yield
        tuples matching ``model.train``'s arguments, so this stays arity-agnostic.
        Relies on the dataset carrying a known cardinality (asserted in ``combine``
        and the per-subject splits) so the progress bar has a total. ``prefix``
        labels the bar (e.g. ``epoch=3/20``)."""
        batches = len(dataset)
        total = 0.0
        for batch in tqdm(dataset, total=batches, desc=f'{prefix} train'.strip(), leave=False):
            total += float(self.model.train(*batch)['loss'])
        return total / batches if batches else 0.0

    def combine(self, datasets: list[tf.data.Dataset]) -> tf.data.Dataset:
        """Merge per-subject datasets for the non-federated loops. Default is
        uniform sampling; override for weighted/interleaved mixing."""
        count = sum([len(ds) for ds in datasets])

        return tf.data.Dataset.sample_from_datasets(datasets)\
               .apply(tf.data.experimental.assert_cardinality(count))

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact (e.g. an AE reconstruction plot)."""
        pass


class AutoencoderTrainer(Trainer):
    """Shared trainer for non-conditional autoencoders (LSTM/GRU/CNN/...).

    Windows the raw ``[BVP, ACC]`` signals from subject-signals and z-score
    normalizes them at load time (so no normalized copy is stored on disk), then
    scores with reconstruction error. Conditional variants subclass this and
    override only ``_windowed`` / ``representative_dataset``.
    """

    primary_metric = 'recon_error'
    default_batch_size = 12

    def __init__(self, model: TrainableModel, window_size: int, shift: int,
                 batch_size: int, train_split: float = 0.975,
                 data_subdir: str = SUBJECTS_SUBDIR):
        self.model = model
        self.window_size = window_size
        self.shift = shift
        self.batch_size = batch_size
        self.train_split = train_split
        self.data_subdir = data_subdir

    def _windowed(self, subject_dir: Path) -> tuple[tf.data.Dataset, int]:
        """One subject's unbatched window tuples. Override to add conditioning."""
        sig_ds, count = windowed_normalized(
            subject_dir.parent, subject_dir.name, self.window_size, self.shift)
        return sig_ds.map(lambda s: (s,)), count

    def subject_datasets(self, data_root, seed):
        data_dir = data_root / self.data_subdir
        subject_dirs = sorted(data_dir.glob('S*'))
        if not subject_dirs:
            raise DatasetUnavailibleError('Signal', data_dir)

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

    def evaluate(self, dataset, prefix=''):
        errors = [self.model.eval(*batch)['error']
                  for batch in tqdm(dataset, total=len(dataset),
                                    desc=f'{prefix} eval'.strip(), leave=False)]
        return {'recon_error': float(tf.reduce_mean(tf.concat(errors, axis=0)))}

    def report(self, result_dir, eval_dataset):
        import matplotlib.pyplot as plt
        for batch in eval_dataset.take(1):
            recon = self.model.eval(*batch)['reconstruction']
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(batch[0][0].numpy())
            axs[0].set_title('Input window [BVP, ACC]')
            axs[1].plot(recon[0].numpy())
            axs[1].set_title('Reconstruction [BVP]')
            fig.savefig(result_dir / 'reconstruction.png')
            print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
            break
