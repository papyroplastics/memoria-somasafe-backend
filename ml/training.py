from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .models.common import Trainer

History = list[tuple[int, float, dict[str, float]]]


def fed_avg(vectors: Sequence[tf.Tensor | np.ndarray],
            sizes: Sequence[int] | None = None) -> np.ndarray:
    """FedAvg of flat parameter vectors, weighted by client dataset size
    (uniform when ``sizes`` is None, as in the backend where submissions carry
    no sample counts). Shared by the simulated ``federated_loop`` and the
    backend aggregation task so simulation matches deployment."""
    stacked = np.stack([np.asarray(vector) for vector in vectors])
    weights = None if sizes is None else np.asarray(sizes, dtype=stacked.dtype)
    return np.average(stacked, axis=0, weights=weights)


def _format(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def normal_loop(trainer: Trainer, train_dataset: tf.data.Dataset,
                eval_dataset: tf.data.Dataset, epochs: int) -> History:
    history: History = []
    for epoch in tqdm(range(epochs), desc="epochs"):
        prefix = f"epoch={epoch + 1}/{epochs}"
        loss = trainer.train_epoch(train_dataset, prefix)
        metrics = trainer.evaluate(eval_dataset, prefix)
        history.append((epoch, loss, metrics))
        print(f"{prefix} loss={loss:.4f} {_format(metrics)}", flush=True)
    return history


def federated_loop(trainer: Trainer, subject_train_datasets: list[tf.data.Dataset],
                   eval_dataset: tf.data.Dataset, local_epochs: int,
                   global_epochs: int, aggregate=fed_avg) -> History:
    model = trainer.model
    sizes = [len(ds) for ds in subject_train_datasets]
    global_weights = model.save()['parameters']

    history: History = []
    for r in tqdm(range(global_epochs), desc="rounds"):
        round_prefix = f"round={r + 1}/{global_epochs}"
        client_weights: list[tf.Tensor] = []
        loss = 0.0
        subjects = tqdm(enumerate(subject_train_datasets),
                        total=len(subject_train_datasets),
                        desc=f"{round_prefix} subjects", leave=False)
        for s, train_ds in subjects:
            model.restore(tf.constant(global_weights))
            for e in tqdm(range(local_epochs),
                          desc=f"{round_prefix} subject={s + 1}/{len(subject_train_datasets)} local",
                          leave=False):
                prefix = f"{round_prefix} subject={s + 1}/{len(subject_train_datasets)} local={e + 1}/{local_epochs}"
                loss = trainer.train_epoch(train_ds, prefix)
            client_weights.append(model.save()['parameters'])

        global_weights = aggregate(client_weights, sizes)
        model.restore(tf.constant(global_weights))

        metrics = trainer.evaluate(eval_dataset, round_prefix)
        history.append((r, loss, metrics))
        print(f"{round_prefix} {_format(metrics)}", flush=True)
    return history
