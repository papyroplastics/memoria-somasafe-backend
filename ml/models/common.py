from numpy import dtype
from abc import ABC, abstractmethod
from typing import Protocol
from pathlib import Path
import hashlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..optimizers import Adam
from ..layers import Dense, relu
from ..metrics import mse_loss, first_difference_loss, reconstruction_error
from ..data import (
    DatasetUnavailibleError, CLEAN_SUBDIR, BVP_RATE,
    windowed_conditional, get_sorted_paths, combine_datasets,
    stacked_signal, norm_stats, window_cond_vectors,
    load_context_norm_params, load_static_norm_params,
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
    infer: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    train: tf.types.experimental.PolymorphicFunction = unbound   # type: ignore
    save: tf.types.experimental.PolymorphicFunction = unbound    # type: ignore
    restore: tf.types.experimental.PolymorphicFunction = unbound # type: ignore

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

    def __init__(self, name: str, batch_size: int, seq_len: int, n_signals: int, n_cond: int,
                 cond_embed_dim, n_outputs, diff_weight, latent_dropout: float,
                 signal_mean, signal_std, cond_mean, cond_std):
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

        # Inputs arrive raw; the model z-scores them so the device and firmware never
        # ship or apply normalization params. Signal params broadcast over (batch, seq).
        self.signal_mean = tf.constant(signal_mean, dtype=tf.float32)
        self.signal_std = tf.constant(signal_std, dtype=tf.float32)
        self.cond_mean = tf.constant(cond_mean, dtype=tf.float32)
        self.cond_std = tf.constant(cond_std, dtype=tf.float32)

        self.cond_dense1 = Dense(n_cond, 32, activation=relu)
        self.cond_dense2 = Dense(32, cond_embed_dim, activation=relu)

    def _embed_cond(self, cond: tf.Tensor) -> tf.Tensor:
        return self.cond_dense2(self.cond_dense1(cond))

    def _drop_latent(self, z: tf.Tensor, training: bool) -> tf.Tensor:
        if training and self.latent_dropout > 0.0:
            return tf.nn.dropout(z, rate=self.latent_dropout)
        return z

    def _bind(self, learning_rate: float, beta1: float, beta2: float, epsilon: float):
        """Bind train/eval/infer/save/restore. Call once all layers exist."""
        self.optimizer = Adam(self.trainable_variables, learning_rate, beta1, beta2, epsilon)
        signature = [
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.cond_shape, dtype=tf.float32),
        ]
        # eval/train z-score raw inputs; infer takes already-normalized inputs and is the
        # only signature exported to the int8 model, so its int8 input calibrates on
        # normalized values (see saving.optimize_saved_model).
        self.eval = tf.function(self.eval_eager, input_signature=signature)
        self.infer = tf.function(self.infer_eager, input_signature=signature)
        self.train = tf.function(self.train_eager, input_signature=signature)
        self._init_save_restore()

    def _forward(self, signal: tf.Tensor, cond: tf.Tensor, training: bool = False) -> tf.Tensor:
        raise NotImplementedError

    def _eval_core(self, signal: tf.Tensor, cond: tf.Tensor):
        """Reconstruction + error from already-normalized signal/cond."""
        reconstruction = self._forward(signal, cond, training=False)
        target = signal[:,:,:self.n_outputs]
        return {'reconstruction': reconstruction,
                'error': reconstruction_error(reconstruction, target)}

    def infer_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        return self._eval_core(signal, cond)

    def eval_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        signal = (signal - self.signal_mean) / self.signal_std
        cond = (cond - self.cond_mean) / self.cond_std
        return self._eval_core(signal, cond)

    def train_eager(self, signal: tf.Tensor, cond: tf.Tensor):
        signal = (signal - self.signal_mean) / self.signal_std
        cond = (cond - self.cond_mean) / self.cond_std
        target = signal[:,:,:self.n_outputs]
        with tf.GradientTape() as tape:
            reconstruction = self._forward(signal, cond, training=True)
            loss = (mse_loss(reconstruction, target)
                    + self.diff_weight * first_difference_loss(reconstruction, target))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


class Trainer(ABC):
    model: TrainableModel
    primary_metric: str
    default_batch_size: int
    batch_size: int
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

    @abstractmethod
    def subject_dataset(self, subject_dir: Path) -> tf.data.Dataset:
        """Returns the data for a single subject"""

    @abstractmethod
    def norm_param_bytes(self) -> bytes:
        """The model's z-score params as LE float32, covered by the server's model
        signature (see ml.payload). Layout is fixed by ``contract_version``; the device
        applies them as ``(x - mean) / std`` before the int8 (non-normalizing) model."""

    @abstractmethod
    def representative_dataset(self, dataset: tf.data.Dataset | None = None, data_root: Path | None = None) -> tf.data.Dataset:
        """Feed-dict stream for the int8 TFLite converter. Loads from data_root when dataset is None."""

    @abstractmethod
    def eval_metrics(self, datapoints: list, outputs: list[dict]) -> dict[str, float]:
        """Metrics relevant to this model type (accuracy, recon error, ...) from the
        aligned lists of evaluated ``datapoints`` (each a full dataset batch tuple) and
        per-datapoint eval-signature ``outputs``. Kept independent of the runtime that
        produced the outputs so both the in-process TF path (``evaluate``) and the
        on-device LiteRT path (``scripts/fed_client.py``) share it; output values may be
        tf tensors or numpy arrays and target tensors are read off ``datapoints``."""

    def evaluate(self, dataset: tf.data.Dataset, prefix: str = '') -> dict[str, float]:
        """Evaluate the model over ``dataset`` and reduce to metrics via ``eval_metrics``.
        ``prefix`` labels the progress bar (e.g. ``epoch=3/20``)."""
        datapoints = list(dataset)
        outputs = [self.model.eval(*dp[:self.n_eval_inputs])
                   for dp in tqdm(datapoints, desc=f'{prefix} eval'.strip(), leave=False)]
        return self.eval_metrics(datapoints, outputs)

    def train_epoch(self, dataset: tf.data.Dataset, prefix: str = '') -> float:
        """One pass over ``dataset``; returns mean training loss. """
        batches = len(dataset)
        total = 0.0
        for batch in tqdm(dataset, total=batches, desc=f'{prefix} train'.strip(), leave=False):
            total += float(self.model.train(*batch)['loss'])
        return total / batches if batches else 0.0

    def report(self, result_dir: Path, eval_dataset: tf.data.Dataset) -> None:
        """Optional model-specific artifact."""
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
            subj_train.append(ds.take(n_train).cache())
            subj_eval.append(ds.skip(n_train).cache())

        return subj_train, combine_datasets(subj_eval).cache()

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
    def __call__(self, data_root: Path, batch_size: int | None = None) -> Trainer: ...


def autoencoder_norm_params(data_root: Path, data_subdir: str = CLEAN_SUBDIR):
    """z-score params baked into an autoencoder so it normalizes its own raw inputs:
    signal per ``[BVP, ACC]`` channel and cond as ``[static(6), context(2)]``."""
    subjects_dir = data_root / data_subdir
    sig_mean, sig_std = norm_stats(subjects_dir)
    stat_mean, stat_std = load_static_norm_params(subjects_dir)
    ctx_mean, ctx_std = load_context_norm_params(subjects_dir)
    cond_mean = np.concatenate([stat_mean, ctx_mean]).astype(np.float32)
    cond_std = np.concatenate([stat_std, ctx_std]).astype(np.float32)
    return sig_mean, sig_std, cond_mean, cond_std

class AutoencoderTrainer(Trainer):

    primary_metric = 'recon_error'
    dataset_tensors = ['signal', 'cond']
    n_eval_inputs = 2
    contract_version = 2   # norm layout: signal mean/std(2 each), cond mean/std(8 each)

    default_batch_size = 12
    default_sample_rate = BVP_RATE
    default_window_size = default_sample_rate * 8
    default_shift = default_sample_rate * 3

    def __init__(self, model: TrainableAutoencoder, window_size: int = default_window_size,
                 shift: int = default_shift, batch_size: int = default_batch_size,
                 data_subdir: str = CLEAN_SUBDIR):
        self.model: TrainableAutoencoder = model # type: ignore
        self.window_size = window_size
        self.shift = shift
        self.batch_size = batch_size
        self.data_subdir = data_subdir

    def norm_param_bytes(self):
        return np.concatenate([
            self.model.signal_mean.numpy(), self.model.signal_std.numpy(),
            self.model.cond_mean.numpy(), self.model.cond_std.numpy(),
        ]).astype('<f4').tobytes()

    def subject_dataset(self, subject_dir):
        return windowed_conditional(subject_dir.parent, subject_dir.name, self.window_size, self.shift)

    def representative_dataset(self, dataset=None, data_root=None):
        # Calibrates the `infer` graph, which takes already-normalized inputs, so yield
        # z-scored signal/cond (matching what the device feeds after normalizing).
        sig_mean = tf.constant(self.model.signal_mean)
        sig_std = tf.constant(self.model.signal_std)
        cond_mean = tf.constant(self.model.cond_mean)
        cond_std = tf.constant(self.model.cond_std)
        if dataset is None:
            if data_root is None:
                raise ValueError("Either dataset or data_root must be passed")

            rng = np.random.default_rng()
            data_dir = data_root / self.data_subdir
            all_signals, all_conds = [], []
            for subject_dir in get_sorted_paths(data_dir):
                sid = subject_dir.name
                raw = stacked_signal(data_dir, sid)
                count = max(0, (len(raw) - self.window_size) // self.shift + 1)
                if count == 0:
                    continue
                cond = window_cond_vectors(data_dir, sid, raw[:, 1], self.window_size, self.shift, count)
                idx = rng.choice(count, size=min(10, count), replace=False)
                all_signals.append(np.stack([raw[i * self.shift : i * self.shift + self.window_size] for i in idx]))
                all_conds.append(cond[idx])
            dataset = tf.data.Dataset.from_tensor_slices((
                np.concatenate(all_signals).astype(np.float32),
                np.concatenate(all_conds).astype(np.float32),
            )).batch(self.batch_size, drop_remainder=True)
        else:
            dataset = dataset.take(150)
        return dataset.map(lambda s, c: {'signal': (s - sig_mean) / sig_std,
                                         'cond': (c - cond_mean) / cond_std})

    def eval_metrics(self, datapoints, outputs):
        errors = np.concatenate([np.asarray(o['error']).reshape(-1) for o in outputs])
        return {'recon_error': float(np.mean(errors))}

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
