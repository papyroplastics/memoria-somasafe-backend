from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common.config import DISABLE_TQDM

from .models.common import Trainer

History = list[tuple[int, float, dict[str, float]]]


def fed_avg(vectors: Sequence[tf.Tensor | np.ndarray],
            sizes: Sequence[int] | None = None) -> np.ndarray:
    stacked = np.stack([np.asarray(vector) for vector in vectors])
    weights = None if sizes is None else np.asarray(sizes, dtype=stacked.dtype)
    return np.average(stacked, axis=0, weights=weights)


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
                eval_dataset: tf.data.Dataset, epochs: int) -> History:
    history: History = []
    for epoch in range(epochs):
        prefix = f"epoch={epoch + 1}/{epochs}"
        loss = train_epoch(trainer, train_dataset, prefix)
        metrics = evaluate(trainer, eval_dataset, prefix)
        history.append((epoch, loss, metrics))
        print(f"{prefix} loss={loss:.4f} {_format(metrics)}", flush=True)
    return history


def federated_loop(trainer: Trainer, subject_train_datasets: list[tf.data.Dataset],
                   eval_dataset: tf.data.Dataset, local_epochs: int,
                   rounds: int, aggregate=fed_avg) -> History:
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

        global_weights = (base + aggregate(client_deltas, sizes)).astype(base.dtype)
        model.restore(tf.constant(global_weights))

        metrics = evaluate(trainer, eval_dataset, round_prefix)
        history.append((r, loss, metrics))
        print(f"{round_prefix} {_format(metrics)}", flush=True)
    return history
