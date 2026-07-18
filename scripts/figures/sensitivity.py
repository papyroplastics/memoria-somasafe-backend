"""Non-IID / clients / LOSO sensitivity sweeps (report Sec. 5.7): simulation sweeps over
`federated_loop` arguments, to show the conclusions do not depend on a lucky
configuration. Three sweeps, each emitting a figure + CSV + companion summary:

  - participants:  final metric vs. number of participating client subjects per round.
  - local-epochs:  final metric vs. local epochs per round.
  - loso:          leave-one-subject-out rotation of the held-out subject; the final metric
                   per fold, reported as mean +/- std (PPG-DaLiA has 15 subjects, so an
                   unseen subject is always held out — the curve is generalization, not
                   memorization).

Like byzantine.py these have to train (they sweep configurations no train.py run
produces), but every run trains a fresh model over the same subject datasets, which are
built once (ml.loading caches them) since they never depend on the weights.

    uv run -m scripts.figures.sensitivity cnn-ae --rounds 5
    uv run -m scripts.figures.sensitivity cnn-ae --sweep loso --loso-folds 5
"""

import argparse

import numpy as np

from common.config import DATASETS_DIR, SEED
from ml.loading import holdout, pool
from ml.model_list import MODELS
from ml.training import federated_loop
from ..common.plots import bar_plot, line_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml


def better_direction(metric: str) -> str:
    return 'lower' if 'error' in metric else 'higher'


def final_metric(key: str, clients: list, eval_dataset, local_epochs: int,
                 rounds: int) -> float:
    """The held-out primary metric after the last round of one federated run. The model
    is rebuilt per run so a loop that mutates the weights never leaks into the next
    configuration; the subject datasets come back from ml.loading's cache."""
    trainer = MODELS[key].build_trainer(DATASETS_DIR)
    history = federated_loop(trainer, clients, eval_dataset, local_epochs, rounds)
    return history[-1][2][trainer.primary_metric]


def sweep_participants(key, subjects, metric, args, report_dir):
    clients, held_out = holdout(subjects, args.eval_subjects)
    eval_dataset = pool(held_out)
    counts = list(range(args.min_participants, len(clients) + 1))
    rows, values = [], []
    for k in counts:
        v = final_metric(key, clients[:k], eval_dataset, args.local_epochs, args.rounds)
        values.append(v)
        rows.append({'participants': k, metric: v})
        print(f"participants={k}: {metric}={v:.6f}")

    line_plot(report_dir / 'participants.png', counts, {metric: values},
              'participating clients', f'final {metric}',
              f'{key} — clients per round')
    write_metrics_csv(rows, report_dir, 'participants.csv')
    write_yaml(report_dir / 'participants.yaml', {
        'shows': f"Sensitivity of {key} to the number of participating client "
                 f"subjects per round.",
        'x_axis': {'name': 'participating clients', 'range': [counts[0], counts[-1]]},
        'y_axis': {'name': f'final {metric} after {args.rounds} rounds',
                   'better': better_direction(metric)},
        'split': {'eval_subjects': args.eval_subjects,
                  'holdout': f'leave-{args.eval_subjects}-subject-out',
                  'local_epochs': args.local_epochs, 'rounds': args.rounds},
        'headline': {'min': min(values), 'max': max(values),
                     'spread': max(values) - min(values)},
        'source': {'seed': SEED, 'reproducible': True},
    })


def sweep_local_epochs(key, subjects, metric, args, report_dir):
    clients, held_out = holdout(subjects, args.eval_subjects)
    eval_dataset = pool(held_out)
    epochs = list(range(1, args.max_local_epochs + 1))
    rows, values = [], []
    for e in epochs:
        v = final_metric(key, clients, eval_dataset, e, args.rounds)
        values.append(v)
        rows.append({'local_epochs': e, metric: v})
        print(f"local_epochs={e}: {metric}={v:.6f}")

    line_plot(report_dir / 'local_epochs.png', epochs, {metric: values},
              'local epochs per round', f'final {metric}',
              f'{key} — local epochs')
    write_metrics_csv(rows, report_dir, 'local_epochs.csv')
    write_yaml(report_dir / 'local_epochs.yaml', {
        'shows': f"Sensitivity of {key} to the number of local epochs per round.",
        'x_axis': {'name': 'local epochs per round', 'range': [1, args.max_local_epochs]},
        'y_axis': {'name': f'final {metric} after {args.rounds} rounds',
                   'better': better_direction(metric)},
        'split': {'clients': len(clients), 'eval_subjects': args.eval_subjects,
                  'holdout': f'leave-{args.eval_subjects}-subject-out',
                  'rounds': args.rounds},
        'headline': {'min': min(values), 'max': max(values),
                     'spread': max(values) - min(values)},
        'source': {'seed': SEED, 'reproducible': True},
    })


def sweep_loso(key, subjects, metric, args, report_dir):
    folds = len(subjects) if args.loso_folds <= 0 else min(args.loso_folds, len(subjects))
    rows, values = [], []
    for i in range(folds):
        clients = [ds for j, ds in enumerate(subjects) if j != i]
        v = final_metric(key, clients, pool([subjects[i]]), args.local_epochs, args.rounds)
        values.append(v)
        rows.append({'held_out_index': i, metric: v})
        print(f"held-out subject #{i}: {metric}={v:.6f}")

    mean, std = float(np.mean(values)), float(np.std(values))
    bar_plot(report_dir / 'loso.png', list(range(folds)), values,
             'held-out subject (fold)', f'final {metric}',
             f'{key} — leave-one-subject-out', mean_line=mean)
    write_metrics_csv(rows, report_dir, 'loso.csv')
    write_yaml(report_dir / 'loso.yaml', {
        'shows': f"Leave-one-subject-out generalization of {key}: the conclusions "
                 f"hold whichever subject is held out.",
        'x_axis': {'name': 'held-out subject (fold)', 'folds': folds},
        'y_axis': {'name': f'final {metric} on that unseen subject after {args.rounds} '
                           f'rounds',
                   'better': better_direction(metric)},
        'split': {'clients_per_fold': len(subjects) - 1, 'eval_subjects': 1,
                  'holdout': 'leave-1-subject-out, rotated',
                  'local_epochs': args.local_epochs, 'rounds': args.rounds},
        'headline': {'mean': mean, 'std': std, 'min': min(values), 'max': max(values)},
        'source': {'seed': SEED, 'reproducible': True},
    })


SWEEPS = {'participants': sweep_participants,
          'local-epochs': sweep_local_epochs,
          'loso': sweep_loso}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to sweep')
    parser.add_argument('--sweep', choices=[*SWEEPS, 'all'], default='all',
                        help='Which sweep to run (default: all)')
    parser.add_argument('--rounds', type=int, default=5, help='Global rounds (default: 10)')
    parser.add_argument('--local-epochs', type=int, default=2,
                        help='Local epochs per round for the non-local-epoch sweeps (default: 2)')
    parser.add_argument('--eval-subjects', type=int, default=2,
                        help='Subjects held out for the participants/local-epochs sweeps (default: 2)')
    parser.add_argument('--min-participants', type=int, default=2,
                        help='Smallest client count in the participants sweep (default: 2)')
    parser.add_argument('--max-local-epochs', type=int, default=5,
                        help='Largest local-epoch count in that sweep (default: 5)')
    parser.add_argument('--loso-folds', type=int, default=0,
                        help='LOSO folds (0 = every subject; default: 0)')
    args = parser.parse_args()

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    subjects = trainer.subject_datasets(DATASETS_DIR)
    report_dir = get_report_dir(args.model, 'sensitivity')

    chosen = list(SWEEPS) if args.sweep == 'all' else [args.sweep]
    for name in chosen:
        print(f"\n=== sweep: {name} ===")
        SWEEPS[name](args.model, subjects, trainer.primary_metric, args, report_dir)


if __name__ == "__main__":
    main()
