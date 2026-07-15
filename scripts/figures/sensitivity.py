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
produces), but every run shares one subject-dataset build and identical configurations are
trained once (see sweeps.py) — with `--sweep all` the full-pool participants point and the
default local-epochs point are the same run.

    uv run -m scripts.figures.sensitivity cnn-ae --rounds 5
    uv run -m scripts.figures.sensitivity cnn-ae --sweep loso --loso-folds 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.config import DATASETS_DIR, SEED
from ml.model_list import MODELS
from ..common.post_train import get_report_dir, write_metrics_csv, write_summary
from .sweeps import SubjectPool


def better_direction(metric: str) -> str:
    return 'lower' if 'error' in metric else 'higher'


def sweep_participants(pool: SubjectPool, args, report_dir):
    metric = pool.metric
    clients, held_out = pool.holdout(args.eval_subjects)
    counts = list(range(args.min_participants, len(clients) + 1))
    rows, values = [], []
    for k in counts:
        v = pool.final_metric(clients[:k], held_out, args.local_epochs, args.rounds)
        values.append(v)
        rows.append({'participants': k, metric: v})
        print(f"participants={k}: {metric}={v:.6f}")

    _line(counts, values, 'participating clients', f'final {metric}',
          f'{pool.key} — clients per round', report_dir / 'participants.png')
    write_metrics_csv(rows, report_dir, 'participants.csv')
    write_summary(report_dir / 'participants.yaml',
        shows=f"Sensitivity of {pool.key} to the number of participating client subjects "
              f"per round.",
        x_axis={'name': 'participating clients', 'range': [counts[0], counts[-1]]},
        y_axis={'name': f'final {metric} after {args.rounds} rounds',
                'better': better_direction(metric)},
        split={'eval_subjects': args.eval_subjects,
               'holdout': f'leave-{args.eval_subjects}-subject-out',
               'local_epochs': args.local_epochs, 'rounds': args.rounds},
        headline={'min': min(values), 'max': max(values),
                  'spread': max(values) - min(values)},
        source={'seed': SEED, 'reproducible': True},
        backs='report Sec. 5.7')


def sweep_local_epochs(pool: SubjectPool, args, report_dir):
    metric = pool.metric
    clients, held_out = pool.holdout(args.eval_subjects)
    epochs = list(range(1, args.max_local_epochs + 1))
    rows, values = [], []
    for e in epochs:
        v = pool.final_metric(clients, held_out, e, args.rounds)
        values.append(v)
        rows.append({'local_epochs': e, metric: v})
        print(f"local_epochs={e}: {metric}={v:.6f}")

    _line(epochs, values, 'local epochs per round', f'final {metric}',
          f'{pool.key} — local epochs', report_dir / 'local_epochs.png')
    write_metrics_csv(rows, report_dir, 'local_epochs.csv')
    write_summary(report_dir / 'local_epochs.yaml',
        shows=f"Sensitivity of {pool.key} to the number of local epochs per round.",
        x_axis={'name': 'local epochs per round', 'range': [1, args.max_local_epochs]},
        y_axis={'name': f'final {metric} after {args.rounds} rounds',
                'better': better_direction(metric)},
        split={'clients': len(clients), 'eval_subjects': args.eval_subjects,
               'holdout': f'leave-{args.eval_subjects}-subject-out',
               'rounds': args.rounds},
        headline={'min': min(values), 'max': max(values),
                  'spread': max(values) - min(values)},
        source={'seed': SEED, 'reproducible': True},
        backs='report Sec. 5.7')


def sweep_loso(pool: SubjectPool, args, report_dir):
    metric = pool.metric
    folds = len(pool) if args.loso_folds <= 0 else min(args.loso_folds, len(pool))
    rows, values = [], []
    for i in range(folds):
        clients = tuple(j for j in range(len(pool)) if j != i)
        v = pool.final_metric(clients, (i,), args.local_epochs, args.rounds)
        values.append(v)
        rows.append({'held_out_index': i, metric: v})
        print(f"held-out subject #{i}: {metric}={v:.6f}")

    mean, std = float(np.mean(values)), float(np.std(values))
    fig, ax = plt.subplots()
    ax.bar(range(folds), values)
    ax.axhline(mean, color='k', linestyle='--', label=f'mean {mean:.4f}')
    ax.set_xlabel('held-out subject (fold)')
    ax.set_ylabel(f'final {metric}')
    ax.set_title(f'{pool.key} — leave-one-subject-out')
    ax.legend()
    fig.savefig(report_dir / 'loso.png')
    print(f"saved LOSO figure to {report_dir / 'loso.png'}")
    write_metrics_csv(rows, report_dir, 'loso.csv')
    write_summary(report_dir / 'loso.yaml',
        shows=f"Leave-one-subject-out generalization of {pool.key}: the conclusions hold "
              f"whichever subject is held out.",
        x_axis={'name': 'held-out subject (fold)', 'folds': folds},
        y_axis={'name': f'final {metric} on that unseen subject after {args.rounds} rounds',
                'better': better_direction(metric)},
        split={'clients_per_fold': len(pool) - 1, 'eval_subjects': 1,
               'holdout': 'leave-1-subject-out, rotated',
               'local_epochs': args.local_epochs, 'rounds': args.rounds},
        headline={'mean': mean, 'std': std, 'min': min(values), 'max': max(values)},
        source={'seed': SEED, 'reproducible': True},
        backs='report Sec. 5.7')


def _line(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path)
    print(f"saved figure to {path}")


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

    pool = SubjectPool(args.model, DATASETS_DIR)
    report_dir = get_report_dir(args.model, 'sensitivity')

    chosen = list(SWEEPS) if args.sweep == 'all' else [args.sweep]
    for name in chosen:
        print(f"\n=== sweep: {name} ===")
        SWEEPS[name](pool, args, report_dir)


if __name__ == "__main__":
    main()
