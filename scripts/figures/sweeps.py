"""Shared helpers for the simulation-based figure scripts (byzantine, sensitivity).

Those two must train — they sweep configurations no single train.py run produces. The
convergence/overlay figures do not; they read a previous run's history (see
plot_convergence.py). What the sweeps share is the expensive part: windowing every
subject into a tf.data pipeline, which is identical across configurations because it
never depends on the model weights. `SubjectPool` builds it once per process and hands
out fresh-weight trainers per run, memoizing runs so a configuration two sweeps have in
common is only trained once.
"""

from pathlib import Path

import tensorflow as tf

from ml.data import combine_datasets
from ml.model_list import MODELS
from ml.models.common import Trainer
from ml.training import History, federated_loop


def build_trainer(key: str, data_dir: Path, batch_size: int | None = None) -> Trainer:
    """Fresh trainer (fresh model + optimizer) for `key`. A new one per run so a loop
    that mutates the model's weights never leaks into the next configuration."""
    return MODELS[key].build_trainer(data_dir, batch_size=batch_size)


def metric_curve(history: History, metric: str) -> list[float]:
    """The primary metric per round, in round order."""
    return [m[metric] for _, _, m in history]


class SubjectPool:
    """Every subject's dataset, built once, plus the runs done against them.

    The datasets are independent of the model weights, so one build serves every
    configuration in a sweep; only the model is rebuilt per run.
    """

    def __init__(self, key: str, data_dir: Path):
        self.key = key
        self.data_dir = data_dir
        trainer = build_trainer(key, data_dir)
        self.metric: str = trainer.primary_metric
        self.subjects: list[tf.data.Dataset] = trainer.all_subject_datasets(data_dir)
        self._runs: dict[tuple, History] = {}

    def __len__(self) -> int:
        return len(self.subjects)

    def eval_dataset(self, indices: tuple[int, ...]) -> tf.data.Dataset:
        return combine_datasets([self.subjects[i] for i in indices])

    def holdout(self, n_eval: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The last `n_eval` subjects held out whole, as index tuples — the same
        leave-N-subject-out split train.py uses."""
        if n_eval < 1:
            raise SystemExit("--eval-subjects must be >= 1")
        if n_eval >= len(self.subjects):
            raise SystemExit(f"--eval-subjects {n_eval} leaves no training subjects "
                             f"({len(self.subjects)} available)")
        indices = tuple(range(len(self.subjects)))
        return indices[:-n_eval], indices[-n_eval:]

    def run(self, clients: tuple[int, ...], held_out: tuple[int, ...], local_epochs: int,
            rounds: int, aggregate=None) -> History:
        """One federated run on a fresh model, memoized on its configuration so two
        sweeps sharing a configuration train it once. Runs with a custom `aggregate` are
        never memoized: the closure carries state (the attack draw) the key can't see.
        """
        key = (clients, held_out, local_epochs, rounds)
        if aggregate is None and key in self._runs:
            print(f"reusing cached run: clients={len(clients)} "
                  f"local_epochs={local_epochs} rounds={rounds}")
            return self._runs[key]

        trainer = build_trainer(self.key, self.data_dir)
        kwargs = {} if aggregate is None else {'aggregate': aggregate}
        history = federated_loop(trainer, [self.subjects[i] for i in clients],
                                 self.eval_dataset(held_out), local_epochs, rounds,
                                 **kwargs)
        if aggregate is None:
            self._runs[key] = history
        return history

    def final_metric(self, clients: tuple[int, ...], held_out: tuple[int, ...],
                     local_epochs: int, rounds: int, aggregate=None) -> float:
        """The held-out primary metric after the last round of one federated run."""
        return metric_curve(self.run(clients, held_out, local_epochs, rounds, aggregate),
                            self.metric)[-1]
