from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common.config import DISABLE_TQDM

from .models.common import Trainer

History = list[tuple[int, float, dict[str, float]]]


def average(vectors: Sequence[tf.Tensor | np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(vector) for vector in vectors]).mean(axis=0)


def weighted_average(vectors: Sequence[tf.Tensor | np.ndarray],
                     sizes: Sequence[int]) -> np.ndarray:
    stacked = np.stack([np.asarray(vector) for vector in vectors])
    return np.average(stacked, axis=0,
                      weights=np.asarray(sizes, dtype=stacked.dtype))


def trimmed_mean(vectors: Sequence[tf.Tensor | np.ndarray],
                 trim: float) -> np.ndarray:
    """Coordinate-wise mean after dropping the `trim` fraction of smallest and largest
    values at each coordinate. `trim` must leave at least one value standing."""
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    stacked = np.sort(np.stack([np.asarray(vector) for vector in vectors]), axis=0)
    k = int(len(stacked) * trim)
    kept = stacked[k:len(stacked) - k] if k else stacked
    return kept.mean(axis=0)


def evaluate(trainer: Trainer, dataset: tf.data.Dataset, prefix: str = '') -> dict[str, float]:
    """Evaluate over ``dataset`` and reduce to metrics via the trainer's ``eval_metrics``.
    ``prefix`` labels the progress bar (e.g. ``epoch=3/20``)."""
    datapoints = list(dataset)
    outputs = [trainer.model.eval(*dp[:trainer.n_eval_inputs])
               for dp in tqdm(datapoints, desc=f'{prefix} eval'.strip(),
                              leave=False, disable=DISABLE_TQDM)]
    return trainer.eval_metrics(datapoints, outputs)


def train_epoch(trainer: Trainer, dataset: tf.data.Dataset, prefix: str = '') -> float:
    """One pass over ``dataset``; returns mean training loss."""
    batches = len(dataset)
    total = 0.0
    for batch in tqdm(dataset, total=batches, desc=f'{prefix} train'.strip(),
                      leave=False, disable=DISABLE_TQDM):
        total += float(trainer.model.train(*batch)['loss'])
    return total / batches if batches else 0.0


def _format(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def normal_loop(trainer: Trainer, train_dataset: tf.data.Dataset,
                eval_dataset: tf.data.Dataset | None, epochs: int) -> History:
    history: History = []
    for epoch in range(epochs):
        prefix = f"epoch={epoch + 1}/{epochs}"
        loss = train_epoch(trainer, train_dataset, prefix)
        metrics = evaluate(trainer, eval_dataset, prefix) if eval_dataset is not None else {}
        history.append((epoch, loss, metrics))
        print(f"{prefix} loss={loss:.4f} {_format(metrics)}", flush=True)
    return history


def federated_loop(trainer: Trainer, subject_train_datasets: list[tf.data.Dataset],
                   eval_dataset: tf.data.Dataset | None, local_epochs: int,
                   rounds: int) -> History:
    model = trainer.model
    sizes = [len(ds) for ds in subject_train_datasets]
    global_weights = model.save()['weights']

    history: History = []
    for r in range(rounds):
        round_prefix = f"round={r + 1}/{rounds}"
        base = np.asarray(global_weights)
        client_deltas: list[np.ndarray] = []
        loss = 0.0
        for s, train_ds in enumerate(subject_train_datasets):
            model.restore(tf.constant(global_weights))
            for e in range(local_epochs):
                prefix = f"{round_prefix} subject={s + 1}/{len(subject_train_datasets)} local={e + 1}/{local_epochs}"
                loss = train_epoch(trainer, train_ds, prefix)
            client_deltas.append(np.asarray(model.save()['weights']) - base)

        global_weights = (base + weighted_average(client_deltas, sizes)).astype(base.dtype)
        model.restore(tf.constant(global_weights))

        metrics = evaluate(trainer, eval_dataset, round_prefix) if eval_dataset is not None else {}
        history.append((r, loss, metrics))
        print(f"{round_prefix} {_format(metrics)}", flush=True)
    return history
