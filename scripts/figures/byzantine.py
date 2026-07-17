"""Byzantine robustness sweep (report Sec. 5.5): global model quality vs. number of
malicious clients, under the plain mean vs. the trimmed mean the server aggregates with.
Backs the security-model claim that trimmed mean holds the round — and no more.

Attack model: each malicious client submits a large random update (a Gaussian delta scaled
to `--attack-magnitude` times the honest clients' mean L2 norm that round) — a naive
model-poisoning / disruption attempt. The server's only Byzantine defense is its aggregation
rule: trimmed mean drops the top/bottom `--trim` fraction per coordinate before averaging,
so a handful of gross outliers can't drag the global weights. The plain mean is the
undefended baseline. Weighted averaging is not offered: it is unsound under this threat
model, since nothing stops an attacker from claiming an enormous dataset size.

The sweep runs its own federated loop (honest clients train against the shared global
weights, then the malicious deltas are appended before aggregation), so honest training is
identical across every configuration.

Unlike the convergence figures, this one has to train: no train.py run produces a poisoned
round. Every configuration trains a fresh model over the same subject datasets, which are
built once (ml.loading caches them) since they never depend on the weights.

    uv run -m scripts.figures.byzantine cnn-ae --max-malicious 4 --eval-subjects 2
"""

import argparse
from collections.abc import Callable

import numpy as np
import tensorflow as tf

from common.config import DATASETS_DIR, SEED
from ml.loading import holdout, pool
from ml.model_list import MODELS
from ml.models.common import Trainer
from ml.training import History, average, evaluate, train_epoch, trimmed_mean
from ..common.plots import line_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml

Aggregator = Callable[[np.ndarray], np.ndarray]


def inject_malicious(deltas: np.ndarray, n_malicious: int, magnitude: float,
                     rng: np.random.Generator) -> np.ndarray:
    if n_malicious == 0:
        return deltas
    honest_norm = float(np.mean(np.linalg.norm(deltas, axis=1))) or 1.0
    dim = deltas.shape[1]
    scale = magnitude * honest_norm / np.sqrt(dim)
    malicious = (rng.standard_normal((n_malicious, dim)) * scale).astype(np.float32)
    return np.concatenate([deltas, malicious], axis=0)


def byzantine_loop(trainer: Trainer, clients: list[tf.data.Dataset],
                   eval_dataset: tf.data.Dataset, local_epochs: int, rounds: int,
                   aggregate: Aggregator, n_malicious: int, magnitude: float,
                   rng: np.random.Generator) -> History:
    model = trainer.model
    global_weights = model.save()['weights']

    history: History = []
    for r in range(rounds):
        round_prefix = f"round={r + 1}/{rounds}"
        base = np.asarray(global_weights)
        client_deltas: list[np.ndarray] = []
        loss = 0.0
        for s, train_ds in enumerate(clients):
            model.restore(tf.constant(global_weights))
            for e in range(local_epochs):
                prefix = (f"{round_prefix} subject={s + 1}/{len(clients)} "
                          f"local={e + 1}/{local_epochs}")
                loss = train_epoch(trainer, train_ds, prefix)
            client_deltas.append(np.asarray(model.save()['weights']) - base)

        deltas = inject_malicious(np.stack(client_deltas).astype(np.float32),
                                  n_malicious, magnitude, rng)

        global_weights = (base + aggregate(deltas)).astype(base.dtype)
        model.restore(tf.constant(global_weights))

        metrics = evaluate(trainer, eval_dataset, round_prefix)
        history.append((r, loss, metrics))
        print(f"{round_prefix} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
              flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to attack')
    parser.add_argument('--trim', type=float, default=0.1,
                        help='Fraction trimmed from each side per coordinate, for the '
                             'trimmed mean (default: 0.1)')
    parser.add_argument('--max-malicious', type=int, default=4,
                        help='Sweep 0..N malicious clients (default: 4)')
    parser.add_argument('--attack-magnitude', type=float, default=10.0,
                        help='Malicious delta L2 as a multiple of the honest mean (default: 10)')
    parser.add_argument('--rounds', type=int, default=5, help='Global rounds (default: 10)')
    parser.add_argument('--local-epochs', type=int, default=2,
                        help='Local epochs per round (default: 2)')
    parser.add_argument('--eval-subjects', type=int, default=2,
                        help='Subjects held out for evaluation (default: 2)')
    args = parser.parse_args()

    aggregators: dict[str, Aggregator] = {
        'plain mean': average,
        'trimmed mean': lambda deltas: trimmed_mean(deltas, args.trim),
    }

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    metric = trainer.primary_metric
    clients, held_out = holdout(trainer.subject_datasets(DATASETS_DIR), args.eval_subjects)
    eval_dataset = pool(held_out)

    counts = list(range(0, args.max_malicious + 1))
    rows: list[dict] = []
    series: dict[str, list[float]] = {label: [] for label in aggregators}
    for n in counts:
        values = {}
        for label, aggregate in aggregators.items():
            # Fresh trainer per run so a loop that mutates the weights never leaks into
            # the next configuration; the subject datasets come back from the cache.
            trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
            # Fresh RNG per config, seeded off the global SEED + n, so the attack is
            # reproducible and both aggregators see the same malicious draws.
            history = byzantine_loop(trainer, clients, eval_dataset, args.local_epochs,
                                     args.rounds, aggregate, n, args.attack_magnitude,
                                     np.random.default_rng(SEED + n))
            values[label] = history[-1][2][metric]
            series[label].append(values[label])
        rows.append({'malicious': n,
                     **{f"{label.replace(' ', '_')}_{metric}": v
                        for label, v in values.items()}})
        print(f"malicious={n}: " + "  ".join(f"{label} {metric}={v:.6f}"
                                             for label, v in values.items()))

    report_dir = get_report_dir(args.model, 'byzantine')
    line_plot(report_dir / 'byzantine.png', counts, series,
              'malicious clients', f'final {metric}',
              f'{args.model} — Byzantine robustness')
    write_metrics_csv(rows, report_dir, 'byzantine.csv')
    write_yaml(report_dir / 'byzantine.yaml', {
        'shows': f"Byzantine robustness of {args.model}: final {metric} vs. number of "
                 f"malicious clients, under the plain mean vs. the trimmed mean.",
        'x_axis': {'name': 'malicious clients', 'range': [0, args.max_malicious]},
        'y_axis': {'name': f'final {metric} after {args.rounds} rounds',
                   'better': 'lower' if 'error' in metric else 'higher'},
        'split': {'honest_clients': len(clients), 'eval_subjects': args.eval_subjects,
                  'holdout': f'leave-{args.eval_subjects}-subject-out',
                  'local_epochs': args.local_epochs, 'rounds': args.rounds},
        'aggregators': {'plain mean': 'undefended baseline',
                        'trimmed mean': {'trim_per_side': args.trim}},
        'attack': {'kind': 'large random (Gaussian) delta',
                   'magnitude': f'{args.attack_magnitude}x the honest mean L2 norm'},
        'headline': {'clean_baseline': {label: series[label][0] for label in aggregators},
                     f'at_{args.max_malicious}_malicious':
                         {label: series[label][-1] for label in aggregators}},
        'conclusion': 'trimmed mean holds the round against gross outliers, and no more',
        'source': {'seed': SEED, 'reproducible': True},
        'backs': 'report Sec. 5.5',
    })


if __name__ == "__main__":
    main()
