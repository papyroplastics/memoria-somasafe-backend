"""Federated convergence (report Sec. 5.2) and the centralized-vs-federated overlay
(Sec. 5.3 — the central thesis claim), both plotted from **previous train.py runs**.

This script trains nothing: it reads the history and manifest each run already wrote to
results/<model>/<loop>/ and fails if they are missing. Produce them with:

    uv run -m scripts.system.train <model> --loop federated --eval-subjects 2
    uv run -m scripts.system.train <model> --loop normal    --eval-subjects 2
    uv run -m scripts.figures.plot_convergence <model>

The federated run alone gives the convergence curve. With a normal run at the same
--eval-subjects, the overlay is drawn too: both loops held out the same subjects and
trained on the same rest, so the curves are comparable and the gap between them is the
claim — FedAvg reaches comparable quality without ever centralizing raw data. The x axis
is the global round / centralized epoch; the compute per step differs (each federated
round runs `local_epochs` local passes).
"""

import argparse

from ml.model_list import MODELS
from ..common.plots import line_plot
from ..common.reports import (
    get_report_dir, read_history_csv, read_run, write_metrics_csv, write_yaml)

# The manifest fields both runs must agree on for the overlay to compare like with like.
COMPARABLE = ('metric', 'eval_subjects', 'train_subjects', 'batch_size', 'dataset_dir')


def better_direction(metric: str) -> str:
    return 'lower' if 'error' in metric or 'loss' in metric else 'higher'


def load_curve(model: str, loop: str) -> tuple[dict, list[float]]:
    """A previous run's manifest and its held-out metric per step."""
    run = read_run(model, loop)
    history = read_history_csv(get_report_dir(model, loop))
    metric = run['metric']
    if not history or any(metric not in row for row in history):
        raise SystemExit(f"the {loop} history for '{model}' has no '{metric}' column; "
                         f"re-run train.py to regenerate it")
    return run, [row[metric] for row in history]


def plot_convergence(model: str, run: dict, values: list[float]) -> None:
    metric = run['metric']
    rounds = list(range(1, len(values) + 1))

    report_dir = get_report_dir(model, 'federated')
    line_plot(report_dir / 'convergence.png', rounds, {metric: values},
              'global round', metric, f'{model} — federated convergence')
    write_metrics_csv([{'round': r, metric: v} for r, v in zip(rounds, values)],
                      report_dir, 'convergence.csv')
    write_yaml(report_dir / 'convergence.yaml', {
        'shows': f"Simulated FedAvg convergence of {model}: the held-out metric improves "
                 f"round over round.",
        'x_axis': {'name': 'global round', 'range': [1, len(values)]},
        'y_axis': {'name': metric, 'better': better_direction(metric)},
        'split': {'clients': run['clients'],
                  'eval_subjects': run['eval_subjects'],
                  'holdout': f"leave-{run['eval_subjects']}-subject-out",
                  'local_epochs': run['local_epochs'],
                  'aggregation': 'uniform-weight FedAvg'},
        'headline': {'first_round': values[0], 'last_round': values[-1],
                     'delta': values[-1] - values[0]},
        'source': {'run': f'results/{model}/federated/run.yaml', 'seed': run['seed'],
                   'reproducible': True},
        'backs': 'report Sec. 5.2',
    })


def plot_overlay(model: str, fed_run: dict, fed_values: list[float],
                 cen_run: dict, cen_values: list[float]) -> None:
    metric = fed_run['metric']
    steps = list(range(1, max(len(cen_values), len(fed_values)) + 1))

    report_dir = get_report_dir(model, 'centralized_vs_federated')
    line_plot(report_dir / 'centralized_vs_federated.png', steps,
              {'centralized': cen_values, 'federated (FedAvg)': fed_values},
              'global round / epoch', metric, f'{model} — centralized vs. federated')
    write_metrics_csv(
        [{'step': s,
          f'centralized_{metric}': cen_values[s - 1] if s <= len(cen_values) else '',
          f'federated_{metric}': fed_values[s - 1] if s <= len(fed_values) else ''}
         for s in steps],
        report_dir, 'centralized_vs_federated.csv')

    caveats = ['per-step compute differs: each federated round runs '
               f"{fed_run['local_epochs']} local passes, a centralized epoch runs one"]
    if len(cen_values) != len(fed_values):
        caveats.append(
            f"the curves have different lengths (centralized {len(cen_values)} epochs, "
            f"federated {len(fed_values)} rounds); compare the final values, not the ends "
            f"of the x axis")

    write_yaml(report_dir / 'centralized_vs_federated.yaml', {
        'shows': f"Centralized vs. federated {metric} for {model} on the same split: "
                 f"FedAvg reaches comparable quality without ever centralizing raw data.",
        'x_axis': {'name': 'global round (federated) / epoch (centralized)',
                   'centralized_range': [1, len(cen_values)],
                   'federated_range': [1, len(fed_values)]},
        'y_axis': {'name': metric, 'better': better_direction(metric)},
        'split': {'eval_subjects': fed_run['eval_subjects'],
                  'holdout': f"leave-{fed_run['eval_subjects']}-subject-out",
                  'train_subjects': fed_run['train_subjects'],
                  'centralized': f"those {fed_run['train_subjects']} subjects pooled",
                  'federated': f"those subjects as {fed_run['clients']} separate clients, "
                               f"{fed_run['local_epochs']} local epoch(s)/round"},
        'headline': {'centralized_final': cen_values[-1],
                     'federated_final': fed_values[-1],
                     'gap_fed_minus_cen': fed_values[-1] - cen_values[-1]},
        'caveats': caveats,
        'source': {'federated_run': f'results/{model}/federated/run.yaml',
                   'centralized_run': f'results/{model}/normal/run.yaml',
                   'seed': fed_run['seed'], 'reproducible': True},
        'backs': 'report Sec. 5.3',
    })


def check_comparable(fed_run: dict, cen_run: dict) -> None:
    """The overlay is only a claim if both runs measured the same thing on the same data."""
    disagree = [f"{f}: federated={fed_run.get(f)!r} normal={cen_run.get(f)!r}"
                for f in COMPARABLE if fed_run.get(f) != cen_run.get(f)]
    if disagree:
        raise SystemExit(
            "the federated and normal runs are not comparable, so the Sec. 5.3 overlay "
            "would be misleading:\n  " + "\n  ".join(disagree) +
            "\nRe-run both with matching settings.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to plot')
    parser.add_argument('--skip-overlay', action='store_true',
                        help='Only plot the federated convergence curve, even if a '
                             'centralized run exists')
    args = parser.parse_args()

    fed_run, fed_values = load_curve(args.model, 'federated')
    plot_convergence(args.model, fed_run, fed_values)

    if args.skip_overlay:
        return

    cen_run, cen_values = load_curve(args.model, 'normal')
    check_comparable(fed_run, cen_run)
    plot_overlay(args.model, fed_run, fed_values, cen_run, cen_values)


if __name__ == "__main__":
    main()
