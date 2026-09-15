"""Non-IID / clients / LOSO sensitivity sweeps (report Sec. 5.7): simulation sweeps over
`federated_loop` arguments — participants per round, local epochs per round, and
leave-one-subject-out rotation — to show the conclusions do not depend on a lucky
configuration. Each sweep trains a fresh model per run over the same subject datasets and
emits a figure, CSV and companion summary."""

import argparse

import numpy as np

from common.config import DATASETS_DIR, SEED
from ml.sources.common import holdout, pool
from ml.model_list import MODELS
from ml.training import federated_loop
from ..common.plots import bar_plot, line_plot
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml


def final_metric(key: str, clients: list, eval_dataset, local_epochs: int,
                 rounds: int) -> float:
    """The held-out primary metric after the last round of one federated run."""
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
        'model': key,
        'metric': metric,
        'eval_subjects': args.eval_subjects,
        'local_epochs': args.local_epochs,
        'rounds': args.rounds,
        'seed': SEED,
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
        'model': key,
        'metric': metric,
        'clients': len(clients),
        'eval_subjects': args.eval_subjects,
        'rounds': args.rounds,
        'seed': SEED,
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
        'model': key,
        'metric': metric,
        'folds': folds,
        'clients_per_fold': len(subjects) - 1,
        'local_epochs': args.local_epochs,
        'rounds': args.rounds,
        'mean': mean, 'std': std,
        'seed': SEED,
    })


SWEEPS = {'participants': sweep_participants,
          'local-epochs': sweep_local_epochs,
          'loso': sweep_loso}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to sweep')
    parser.add_argument('--sweep', choices=[*SWEEPS, 'all'], default='all', help='Which sweep to run')
    parser.add_argument('--rounds', type=int, default=5, help='Global rounds')
    parser.add_argument('--local-epochs', type=int, default=2,
                        help='Local epochs per round for the non-local-epoch sweeps')
    parser.add_argument('--eval-subjects', type=int, default=2,
                        help='Subjects held out for the participants/local-epochs sweeps')
    parser.add_argument('--min-participants', type=int, default=2,
                        help='Smallest client count in the participants sweep')
    parser.add_argument('--max-local-epochs', type=int, default=5,
                        help='Largest local-epoch count in that sweep')
    parser.add_argument('--loso-folds', type=int, default=0, help='LOSO folds (0 = every subject)')
    args = parser.parse_args()

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    subjects = trainer.subject_datasets()
    report_dir = get_report_dir(args.model, 'sensitivity')

    chosen = list(SWEEPS) if args.sweep == 'all' else [args.sweep]
    for name in chosen:
        print(f"\n=== sweep: {name} ===")
        SWEEPS[name](args.model, subjects, trainer.primary_metric, args, report_dir)


if __name__ == "__main__":
    main()
