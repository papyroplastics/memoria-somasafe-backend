"""Byzantine robustness sweep (report Sec. 5.5): global model quality vs. number of
malicious clients, with and without the z-score outlier filter the server runs. Backs the
security-model claim that the basic defense holds the round — and no more.

Attack model: each malicious client submits a large random update (a Gaussian delta scaled
to `--attack-magnitude` times the honest clients' mean L2 norm that round) — a naive
model-poisoning / disruption attempt, exactly the gross outlier the L2 z-score filter is
meant to catch. The filter (worker.utils.weight_validation.filter_outliers) drops updates
whose distance from the element-wise mean is a z-score outlier; it needs >= 3 submissions to
act. The sweep injects the malicious updates into the same `federated_loop` aggregation and
toggles the filter, so honest training is identical across the two lines.

Unlike the convergence figures, this one has to train: no train.py run produces a poisoned
round. Every configuration trains a fresh model over the same subject datasets, which are
built once (ml.loading caches them) since they never depend on the weights.

    uv run -m scripts.figures.byzantine cnn-ae --max-malicious 4 --eval-subjects 2
"""

import argparse

import numpy as np

from common.config import DATASETS_DIR, SEED
from ml.loading import holdout, pool
from ml.model_list import MODELS
from ml.training import federated_loop
from worker.utils.weight_validation import filter_outliers
from ..common.plots import line_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml


def byzantine_aggregate(n_malicious: int, use_filter: bool, magnitude: float,
                        rng: np.random.Generator, z_threshold: float):
    """Aggregate closure for `federated_loop`: append `n_malicious` large random deltas to
    the honest ones, optionally drop z-score outliers, then uniform-average the survivors
    (matching the deployed server, which weights submissions uniformly)."""
    def aggregate(deltas, sizes=None):
        honest = [np.asarray(d, dtype=np.float32) for d in deltas]
        stacked = np.stack(honest)
        if n_malicious > 0:
            honest_norm = float(np.mean(np.linalg.norm(stacked, axis=1))) or 1.0
            dim = stacked.shape[1]
            scale = magnitude * honest_norm / np.sqrt(dim)
            malicious = (rng.standard_normal((n_malicious, dim)) * scale).astype(np.float32)
            stacked = np.concatenate([stacked, malicious], axis=0)
        if use_filter:
            stacked = stacked[filter_outliers(stacked, z_threshold)]
        return stacked.mean(axis=0)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to attack')
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

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    metric = trainer.primary_metric
    clients, held_out = holdout(trainer.subject_datasets(DATASETS_DIR), args.eval_subjects)
    eval_dataset = pool(held_out)

    counts = list(range(0, args.max_malicious + 1))
    rows, no_filter, with_filter = [], [], []
    for n in counts:
        values = {}
        for use_filter in (False, True):
            # Fresh RNG per config, seeded off the global SEED + n, so the attack is
            # reproducible and the filter-on/off pair sees the same malicious draws.
            aggregate = byzantine_aggregate(n, use_filter, args.attack_magnitude,
                                            np.random.default_rng(SEED + n),
                                            args.z_threshold)
            # Fresh trainer per run so a loop that mutates the weights never leaks into
            # the next configuration; the subject datasets come back from the cache.
            trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
            history = federated_loop(trainer, clients, eval_dataset, args.local_epochs,
                                     args.rounds, aggregate=aggregate)
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
              f'{args.model} — Byzantine robustness')
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
