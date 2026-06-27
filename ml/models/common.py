from abc import ABC, abstractmethod
from typing import Protocol
from pathlib import Path
import hashlib
import tensorflow as tf
from tqdm import tqdm

from ..optimizers import Adam
from ..layers import Dense
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..data import (
    DatasetUnavailibleError, SUBJECTS_SUBDIR,\
    windowed_conditional, get_sorted_paths, combine_datasets
)

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
    """Conditional reconstruction autoencoder base.

    Subclasses build their encoder/decoder layers and implement ``_forward(signal,
    cond, training)``; the train/eval bodies, signature binding, conditioning
    embedding, latent dropout and Adam optimizer are shared. The encoder sees
    ``n_signals`` channels (``[BVP, ACC]``) but the decoder only reconstructs the
    first ``n_outputs`` (BVP) — ACC is exogenous context, not part of the anomaly
    score. Every model is conditioned on a single ``cond`` vector (z-scored
    demographics + a causal activity context); the embedding here feeds the
    subclass's conditioning mechanism (FiLM for the CNN, latent fusion for the
    RNNs). Latent dropout pushes the decoder to rely on the condition rather than
    copy the input through the bottleneck. The objective is reconstruction MSE plus
    a first-difference term that penalizes flat output.
    """

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int,
                 n_cond: int, cond_embed_dim: int = 16, n_outputs: int = 1,
                 diff_weight: float = 1.0, latent_dropout: float = 0.1):
        super().__init__(name=name)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals
        self.n_cond = n_cond
        self.cond_embed_dim = cond_embed_dim
        self.n_outputs = n_outputs
        self.diff_weight = diff_weight
        self.latent_dropout = latent_dropout
        self.signal_shape = (batch_size, seq_len, n_signals)
        self.cond_shape = (batch_size, n_cond)

        self.cond_dense1 = Dense(n_cond, 32, activation=tf.nn.relu)
        self.cond_dense2 = Dense(32, cond_embed_dim, activation=tf.nn.relu)

    def _embed_cond(self, cond: tf.Tensor) -> tf.Tensor:
        return self.cond_dense2(self.cond_dense1(cond))

    def _drop_latent(self, z: tf.Tensor, training: bool) -> tf.Tensor:
        if training and self.latent_dropout > 0.0:
            return tf.nn.dropout(z, rate=self.latent_dropout)
        return z

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        """Bind train/eval/save/restore. Call once all layers exist."""
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)
        signature = [
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.cond_shape, dtype=tf.float32),
        ]
        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=signature)
        self._init_save_restore()

    def _forward(self, signal: tf.Tensor, cond: tf.Tensor, training: bool = False) -> tf.Tensor:
        raise NotImplementedError

    def eval_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        reconstruction = self._forward(signal, cond, training=False)
        target = signal[:,:,:self.n_outputs]
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, target)}

    def train_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        target = signal[:,:,:self.n_outputs]
        with tf.GradientTape() as tape:
            reconstruction = self._forward(signal, cond, training=True)
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
    data_subdir: str

    @abstractmethod
    def subject_dataset(self, subject_dir: Path) -> tf.data.Dataset:
        """Returns the data for a single subject"""

    @abstractmethod
    def representative_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Feed-dict stream for the int8 TFLite converter."""

    @abstractmethod
    def evaluate(self, dataset: tf.data.Dataset, prefix: str = '') -> dict[str, float]:
        """Metrics relevant to this model type (accuracy, recon error, ...).
        ``prefix`` labels the progress bar (e.g. ``epoch=3/20``)."""

    def train_epoch(self, dataset: tf.data.Dataset, prefix: str = '') -> float:
        """One pass over ``dataset``; returns mean training loss. """
        batches = len(dataset)
        total = 0.0
        for batch in tqdm(dataset, total=batches, desc=f'{prefix} train'.strip(), leave=False):
            total += float(self.model.train(*batch)['loss'])
        return total / batches if batches else 0.0

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact (e.g. an AE reconstruction plot)."""
        pass

    def subject_datasets(
            self, data_root: Path, train_split: float
        ) -> tuple[list[tf.data.Dataset], tf.data.Dataset]:
        """Per-subject datasets for federated trainig loop"""

        data_dir = data_root / self.data_subdir
        subject_dirs = get_sorted_paths(data_dir)
        if not subject_dirs:
            raise DatasetUnavailibleError(data_dir)

        subj_train, subj_eval = [], []
        for d in subject_dirs:
            ds = self.subject_dataset(d)
            ds = ds.shuffle(len(ds)).batch(self.batch_size, drop_remainder=True)

            n_train = int(len(ds) * train_split)
            subj_train.append(ds.take(n_train))
            subj_eval.append(ds.skip(n_train))

        return subj_train, combine_datasets(subj_eval)

    def combined_datasets(
            self, data_root: Path, train_split: float
        ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Joint subject dataset for normal training loop"""

        data_dir = data_root / self.data_subdir
        subject_dirs = get_sorted_paths(data_dir)
        if not subject_dirs:
            raise DatasetUnavailibleError(data_dir)

        subject_datasets = [
            self.subject_dataset(d)
            for d in subject_dirs
        ]

        ds = combine_datasets([
                d.shuffle(len(d), reshuffle_each_iteration=False)
                for d in subject_datasets
            ]).batch(self.batch_size, drop_remainder=True)

        n_train = int(len(ds) * train_split)

        return ds.take(n_train).cache(), ds.skip(n_train).cache()

class TrainerBuilder(Protocol):
    def __call__(self, batch_size: int | None = None) -> Trainer: ...

class AutoencoderTrainer(Trainer):
    """Shared trainer for the (conditional) autoencoders (LSTM/GRU/CNN/...).

    Windows the raw ``[BVP, ACC]`` signals from subject-signals and z-score
    normalizes them at load time (so no normalized copy is stored on disk), pairs
    each window with its conditioning vector (demographics + activity context), and
    scores with reconstruction error.
    """

    primary_metric = 'recon_error'
    default_batch_size = 12

    def __init__(self, model: TrainableModel, window_size: int, shift: int,
                 batch_size: int, data_subdir: str = SUBJECTS_SUBDIR):
        self.model = model
        self.window_size = window_size
        self.shift = shift
        self.batch_size = batch_size
        self.data_subdir = data_subdir

    def subject_dataset(self, subject_dir):
        return windowed_conditional(subject_dir.parent, subject_dir.name, self.window_size, self.shift)

    def representative_dataset(self, dataset):
        return dataset.take(10).map(lambda s, c: {'signal': s, 'cond': c})

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
