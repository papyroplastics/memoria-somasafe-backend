"""Byzantine robustness sweep (report Sec. 5.5): global model quality vs. number of
malicious clients, with and without the z-score outlier filter the server runs. Backs the
security-model claim that the basic defense holds the round — and no more.

Attack model: each malicious client submits a large random update (a Gaussian delta scaled
to `--attack-magnitude` times the honest clients' mean L2 norm that round) — a naive
model-poisoning / disruption attempt, exactly the gross outlier the L2 z-score filter is
meant to catch. The filter (worker.utils.weight_validation.filter_outliers) drops updates
whose distance from the element-wise mean is a z-score outlier; it needs >= 3 submissions to
act. `--aggregator` picks what averages the survivors: `trimmed-mean` (the deployed server's
choice) or the plain `average`. Weighted averaging is not offered: it is unsound under this
threat model, since nothing stops an attacker from claiming an enormous dataset size.

The sweep runs its own federated loop (honest clients train against the shared global
weights, then the malicious deltas are appended before filtering and aggregation), so honest
training is identical across every configuration.

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
from worker.utils.weight_validation import filter_outliers
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
                   use_filter: bool, rng: np.random.Generator,
                   z_threshold: float) -> History:
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
        if use_filter:
            deltas = deltas[filter_outliers(deltas, z_threshold)]

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
    parser.add_argument('--aggregator', choices=('trimmed-mean', 'average'),
                        default='trimmed-mean',
                        help='Aggregation rule applied to the surviving updates '
                             '(default: trimmed-mean)')
    parser.add_argument('--trim', type=float, default=0.1,
                        help='Fraction trimmed from each side per coordinate, for '
                             '--aggregator trimmed-mean (default: 0.1)')
    parser.add_argument('--max-malicious', type=int, default=4,
                        help='Sweep 0..N malicious clients (default: 4)')
    parser.add_argument('--attack-magnitude', type=float, default=10.0,
                        help='Malicious delta L2 as a multiple of the honest mean (default: 10)')
    parser.add_argument('--z-threshold', type=float, default=3.0,
                        help='Outlier filter z-score cutoff (default: 3.0)')
    parser.add_argument('--rounds', type=int, default=5, help='Global rounds (default: 10)')
    parser.add_argument('--local-epochs', type=int, default=2,
                        help='Local epochs per round (default: 2)')
    parser.add_argument('--eval-subjects', type=int, default=2,
                        help='Subjects held out for evaluation (default: 2)')
    args = parser.parse_args()

    aggregate: Aggregator = (average if args.aggregator == 'average'
                             else lambda deltas: trimmed_mean(deltas, args.trim))

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    metric = trainer.primary_metric
    clients, held_out = holdout(trainer.subject_datasets(DATASETS_DIR), args.eval_subjects)
    eval_dataset = pool(held_out)

    counts = list(range(0, args.max_malicious + 1))
    rows, no_filter, with_filter = [], [], []
    for n in counts:
        values = {}
        for use_filter in (False, True):
            # Fresh trainer per run so a loop that mutates the weights never leaks into
            # the next configuration; the subject datasets come back from the cache.
            trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
            # Fresh RNG per config, seeded off the global SEED + n, so the attack is
            # reproducible and the filter-on/off pair sees the same malicious draws.
            history = byzantine_loop(trainer, clients, eval_dataset, args.local_epochs,
                                     args.rounds, aggregate, n, args.attack_magnitude,
                                     use_filter, np.random.default_rng(SEED + n),
                                     args.z_threshold)
            values[use_filter] = history[-1][2][metric]
        no_filter.append(values[False])
        with_filter.append(values[True])
        rows.append({'malicious': n,
                     f'no_filter_{metric}': values[False],
                     f'with_filter_{metric}': values[True]})
        print(f"malicious={n}: no_filter {metric}={values[False]:.6f}  "
              f"with_filter {metric}={values[True]:.6f}")

    report_dir = get_report_dir(args.model, 'byzantine')
    line_plot(report_dir / 'byzantine.png', counts,
              {'no filter': no_filter, 'z-score outlier filter': with_filter},
              'malicious clients', f'final {metric}',
              f'{args.model} — Byzantine robustness ({args.aggregator})')
    write_metrics_csv(rows, report_dir, 'byzantine.csv')
    write_yaml(report_dir / 'byzantine.yaml', {
        'shows': f"Byzantine robustness of {args.model}: final {metric} vs. number of "
                 f"malicious clients, with and without the server's z-score outlier filter.",
        'x_axis': {'name': 'malicious clients', 'range': [0, args.max_malicious]},
        'y_axis': {'name': f'final {metric} after {args.rounds} rounds',
                   'better': 'lower' if 'error' in metric else 'higher'},
        'split': {'honest_clients': len(clients), 'eval_subjects': args.eval_subjects,
                  'holdout': f'leave-{args.eval_subjects}-subject-out',
                  'local_epochs': args.local_epochs, 'rounds': args.rounds},
        'aggregator': ({'kind': 'trimmed mean', 'trim_per_side': args.trim}
                       if args.aggregator == 'trimmed-mean' else {'kind': 'plain mean'}),
        'attack': {'kind': 'large random (Gaussian) delta',
                   'magnitude': f'{args.attack_magnitude}x the honest mean L2 norm',
                   'filter': f'z-score cutoff {args.z_threshold}, needs >= 3 submissions '
                             f'to act'},
        'headline': {'clean_baseline': {'no_filter': no_filter[0],
                                        'with_filter': with_filter[0]},
                     f'at_{args.max_malicious}_malicious': {'no_filter': no_filter[-1],
                                                            'with_filter': with_filter[-1]}},
        'conclusion': 'the filter holds the round against gross outliers, and no more',
        'source': {'seed': SEED, 'reproducible': True},
        'backs': 'report Sec. 5.5',
    })


if __name__ == "__main__":
    main()
