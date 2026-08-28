"""
Train a SomaSafe model with a chosen loop on the low-activity dataset, holding out whole
subjects for evaluation. Weights and serving artifacts go to shared/gen/models/<model>;
the history, plots, run manifest and eval report to results/<model>/<loop>.
"""

import argparse
import re
from pathlib import Path
import tensorflow as tf

from ml.dataset_list import TRAINING_DATASET
from ml.loading import pool
from ml.models.common import Trainer
from ml.saving import load_weights, save_artifacts, weights_path
from ml.training import normal_loop, federated_loop, History
from ml.model_list import MODELS
from common.config import MODELS_DIR, DATASETS_DIR, SEED
from ..common.plots import plot_history
from ..common.reports import RUN_MANIFEST, get_report_dir, loop_dir, write_history_csv, write_yaml

LOOP_OPTIONS = ['normal', 'federated']


def parse_eval_selection(value: str, sids: list[str]) -> list[str]:
    value = value.strip()
    if value == 'none':
        return []
    if re.fullmatch(r'\d+-\d+', value):
        lo, hi = (int(x) for x in value.split('-'))
        ids = {f'S{i}' for i in range(lo, hi + 1)}
    else:
        ids = {f'S{int(i)}' for i in value.split(',')}
    missing = ids - set(sids)
    if missing:
        raise SystemExit(f"eval subjects {sorted(missing)} not found among {sids}")
    return [s for s in sids if s in ids]


def run_loop(trainer: Trainer, loop: str, eval_ids: list[str],
             steps: int, local_epochs: int
             ) -> tuple[History, tf.data.Dataset, list[str], list[str]]:
    datasets = trainer.subject_datasets()
    sids = trainer.subject_ids()
    held = set(eval_ids)
    train_subjects = [ds for ds, sid in zip(datasets, sids) if sid not in held]
    held_out = [ds for ds, sid in zip(datasets, sids) if sid in held]
    if not train_subjects:
        raise SystemExit("the eval selection leaves no training subjects")
    eval_dataset = pool(held_out) if held_out else None

    if loop == 'normal':
        history = normal_loop(trainer, pool(train_subjects), eval_dataset, steps)

    elif loop == 'federated':
        history = federated_loop(trainer, train_subjects, eval_dataset, local_epochs, steps)

    else:
        raise Exception(f"Invalid loop type: {loop}")

    train_ids = [sid for sid in sids if sid not in held]
    return history, eval_dataset, train_ids, [sid for sid in sids if sid in held]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to train')
    parser.add_argument('--loop', choices=LOOP_OPTIONS, default='normal', help='Training loop')
    parser.add_argument('--eval-subjects', default='14-15',
                        help="Subjects held out whole for evaluation: an id N, an inclusive "
                             "range 'n-m', a list 'i,j,k', or 'none' to train on everyone "
                             "and skip evaluation")
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for the normal loop')
    parser.add_argument('--local-epochs', type=int, default=2, help='Local epochs per round (federated)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the model default batch size')
    parser.add_argument('--tag', default=None,
                        help="Suffix for this run's weights, artifacts and results "
                             'directory, so it does not clobber the canonical run')
    parser.add_argument('--no-tflite', action='store_true',
                        help='Skip the .tflite exports and write only the weights')
    parser.add_argument('--load-weights', action='store_true',
                        help='Keep training the weights already saved under --tag instead '
                             'of starting from a fresh model')
    parser.add_argument('--dataset-dir', type=Path, default=DATASETS_DIR,
                        help='Dataset directory to train on, e.g. one holding a '
                             "teacher's distilled labels")
    args = parser.parse_args()

    data_dir = args.dataset_dir

    result_dir = MODELS_DIR / args.model
    result_dir.mkdir(parents=True, exist_ok=True)

    report_dir = get_report_dir(args.model, loop_dir(args.loop, args.tag))

    trainer = MODELS[args.model].build_trainer(data_dir, batch_size=args.batch_size)

    if args.load_weights:
        source = weights_path(result_dir, args.tag)
        if not source.exists():
            raise SystemExit(f"no weights to continue from at {source}")
        trainer.model.restore(load_weights(source))
        print(f"Loaded weights from {source}")

    eval_ids = parse_eval_selection(args.eval_subjects, trainer.subject_ids())
    history, eval_dataset, train_ids, held_ids = run_loop(
        trainer, args.loop, eval_ids, args.epochs, args.local_epochs)

    batch_size = trainer.model.batch_size

    postfix = f'_{args.tag}' if args.tag else ''
    save_artifacts(trainer, result_dir, postfix, tflite=not args.no_tflite)

    if args.epochs == 0:
        exit()

    write_history_csv(history, report_dir)
    if eval_dataset is not None:
        plot_history(history, trainer.primary_metric, report_dir)
        trainer.report(report_dir, eval_dataset)

    _, final_loss, final_metrics = history[-1]
    write_yaml(report_dir / RUN_MANIFEST, {
        'model': args.model,
        'loop': args.loop,
        'tag': args.tag,
        'metric': trainer.primary_metric,
        'epochs': args.epochs,
        'step_unit': 'round' if args.loop == 'federated' else 'epoch',
        'local_epochs': args.local_epochs if args.loop == 'federated' else None,
        'clients': len(train_ids) if args.loop == 'federated' else None,
        'train_subjects': train_ids,
        'eval_subjects': held_ids,
        'batch_size': batch_size,
        'dataset': TRAINING_DATASET,
        'dataset_dir': args.dataset_dir,
        'seed': SEED,
        'history': 'training.csv',
        'final': {'loss': final_loss, **final_metrics},
    })
