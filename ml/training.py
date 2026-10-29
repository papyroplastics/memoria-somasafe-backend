from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from common.config import DISABLE_TQDM

from .aggregation import trimmed_mean
from .models.common import Trainer

History = list[tuple[int, float, dict[str, float]]]
Aggregator = Callable[[np.ndarray, Sequence[int]], np.ndarray]

AGGREGATIONS = ('weighted', 'mean', 'trimmed')


def average(vectors: Sequence[tf.Tensor | np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(vector) for vector in vectors]).mean(axis=0)


def weighted_average(vectors: Sequence[tf.Tensor | np.ndarray],
                     sizes: Sequence[int]) -> np.ndarray:
    stacked = np.stack([np.asarray(vector) for vector in vectors])
    return np.average(stacked, axis=0,
                      weights=np.asarray(sizes, dtype=stacked.dtype))


def aggregator(name: str, trim: float = 0.2) -> Aggregator:
    if name == 'weighted':
        return lambda deltas, sizes: weighted_average(deltas, sizes)
    if name == 'mean':
        return lambda deltas, _: average(deltas)
    if name == 'trimmed':
        return lambda deltas, _: trimmed_mean(deltas, trim)
    raise ValueError(f"unknown aggregation {name!r}, expected one of {AGGREGATIONS}")


def evaluate(trainer: Trainer, dataset: tf.data.Dataset, prefix: str = '') -> dict[str, float]:
    datapoints = list(dataset)
    outputs = [trainer.model.eval(*dp[:trainer.n_eval_inputs])
               for dp in tqdm(datapoints, desc=f'{prefix} eval'.strip(),
                              leave=False, disable=DISABLE_TQDM)]
    return trainer.eval_metrics(datapoints, outputs)


def train_epoch(trainer: Trainer, dataset: tf.data.Dataset, prefix: str = '') -> float:
    batches = len(dataset)
    total = 0.0
    for batch in tqdm(dataset, total=batches, desc=f'{prefix} train'.strip(),
                      leave=False, disable=DISABLE_TQDM):
        total += float(trainer.model.train(*batch)['loss'])
    return total / batches if batches else 0.0


def local_update(trainer: Trainer, global_weights: np.ndarray, dataset: tf.data.Dataset,
                 local_epochs: int, prefix: str = '') -> tuple[np.ndarray, float]:
    model = trainer.model
    model.restore(tf.constant(global_weights))
    model.reset_optimizer()
    loss = 0.0
    for e in range(local_epochs):
        loss = train_epoch(trainer, dataset, f"{prefix} local={e + 1}/{local_epochs}")
    return np.asarray(model.save()['weights']) - global_weights, loss


def _format(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def normal_loop(trainer: Trainer, train_dataset: tf.data.Dataset,
                eval_dataset: tf.data.Dataset | None, epochs: int) -> History:
    history: History = []
    try:
        for epoch in range(epochs):
            prefix = f"epoch={epoch + 1}/{epochs}"
            loss = train_epoch(trainer, train_dataset, prefix)
            metrics = evaluate(trainer, eval_dataset, prefix) if eval_dataset is not None else {}
            history.append((epoch, loss, metrics))
            print(f"{prefix} loss={loss:.4f} {_format(metrics)}", flush=True)
    except KeyboardInterrupt:
        pass

    return history


def federated_loop(trainer: Trainer, subject_train_datasets: list[tf.data.Dataset],
                   eval_dataset: tf.data.Dataset | None, local_epochs: int,
                   rounds: int, aggregate: Aggregator | None = None) -> History:
    aggregate = aggregate or aggregator('weighted')
    model = trainer.model
    sizes = [len(ds) for ds in subject_train_datasets]
    global_weights = np.asarray(model.save()['weights'])

    history: History = []
    try:
        for r in range(rounds):
            round_prefix = f"round={r + 1}/{rounds}"
            client_deltas: list[np.ndarray] = []
            losses: list[float] = []
            for s, train_ds in enumerate(subject_train_datasets):
                prefix = f"{round_prefix} subject={s + 1}/{len(subject_train_datasets)}"
                delta, loss = local_update(trainer, global_weights, train_ds, local_epochs, prefix)
                client_deltas.append(delta)
                losses.append(loss)

            global_weights = (global_weights + aggregate(np.stack(client_deltas), sizes)
                              ).astype(global_weights.dtype)
            model.restore(tf.constant(global_weights))

            metrics = evaluate(trainer, eval_dataset, round_prefix) if eval_dataset is not None else {}
            history.append((r, float(np.mean(losses)), metrics))
            print(f"{round_prefix} {_format(metrics)}", flush=True)
    except KeyboardInterrupt:
        pass

    return history


@dataclass(frozen=True)
class SecureConfig:
    session_size: int = 10
    min_members: int = 3
    trim: float = 0.2
    clip: float = 1.0
    clients_per_round: int | None = None
    secure: bool = True


def arrival_order(honest: list[int], attackers: list[int], session_size: int,
                  rng: np.random.Generator) -> list[int]:
    honest = [int(i) for i in rng.permutation(honest)]
    n_sessions = max(1, (len(honest) + len(attackers)) // session_size)
    slots: list[list[int]] = [[] for _ in range(n_sessions)]
    for j, a in enumerate(attackers):
        slots[j % n_sessions].append(a)
    order: list[int] = []
    for slot in slots:
        fill = max(session_size - len(slot), 0)
        order.extend(slot + honest[:fill])
        honest = honest[fill:]
    return order + honest


def sessions_of(order: list[int], size: int, min_members: int) -> list[list[int]]:
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    return [chunk for chunk in chunks if len(chunk) >= min_members]


def secure_loop(trainer: Trainer, clients: list[tf.data.Dataset],
                eval_dataset: tf.data.Dataset, local_epochs: int, rounds: int,
                config: SecureConfig, malicious_sessions: float = 0.0,
                seed: int = 0) -> History:
    rng = np.random.default_rng(seed)
    model = trainer.model
    global_weights = np.asarray(model.save()['weights'])

    n_attackers = round(malicious_sessions * len(clients) / config.session_size)
    malicious = set(int(i) for i in rng.permutation(len(clients))[:n_attackers])

    history: History = []
    for r in range(rounds):
        round_prefix = f"round={r + 1}/{rounds}"
        k = config.clients_per_round or len(clients)
        participants = [int(i) for i in rng.permutation(len(clients))[:k]]
        honest_ids = [i for i in participants if i not in malicious]
        attacker_ids = [i for i in participants if i in malicious]

        deltas: dict[int, np.ndarray] = {}
        losses: list[float] = []
        for n, i in enumerate(honest_ids):
            prefix = f"{round_prefix} client={n + 1}/{len(honest_ids)}"
            delta, loss = local_update(trainer, global_weights, clients[i], local_epochs, prefix)
            deltas[i] = delta
            losses.append(loss)

        honest = np.stack([deltas[i] for i in honest_ids]) if honest_ids else np.zeros((0, len(global_weights)), np.float32)
        clipped_fraction = float(np.mean(np.abs(honest) > config.clip)) if honest_ids else 0.0
        for i in deltas:
            deltas[i] = np.clip(deltas[i], -config.clip, config.clip).astype(np.float32)

        def garbage() -> np.ndarray:
            return rng.uniform(-config.clip, config.clip, len(global_weights)).astype(np.float32)

        if config.secure:
            order = arrival_order(honest_ids, attacker_ids, config.session_size, rng)
            groups = sessions_of(order, config.session_size, config.min_members)
            rows = [garbage() if malicious.intersection(group)
                    else np.mean([deltas[i] for i in group], axis=0) for group in groups]
        else:
            rows = [deltas[i] if i in deltas else garbage() for i in participants]

        if rows:
            global_weights = (global_weights + trimmed_mean(rows, config.trim)
                              ).astype(global_weights.dtype)
        model.restore(tf.constant(global_weights))

        metrics = evaluate(trainer, eval_dataset, round_prefix)
        metrics.update(rows=len(rows), clipped=clipped_fraction)
        history.append((r, float(np.mean(losses)) if losses else 0.0, metrics))
        print(f"{round_prefix} {_format(metrics)}", flush=True)

    return history
