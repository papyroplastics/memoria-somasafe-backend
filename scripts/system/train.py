"""
Train a SomaSafe model with a chosen training loop. Serving artifacts
(trainable/quantized .tflite) go to shared/gen/models/<model>; the training
history, plot, run manifest and eval report go to results/<model>/<loop>.

Both loops hold out whole subjects (--eval-subjects: an id, an id range, a list, or none)
and score on them, so a centralized and a federated run at the same --eval-subjects train on
the same data and are directly comparable — that overlay is what scripts.figures.plot_convergence
draws from the manifests this writes. The resolved held-out ids are recorded, not just a count.
"""

import argparse
import re
from pathlib import Path
import tensorflow as tf

from ml.loading import pool, subject_dirs
from ml.models.common import Trainer
from ml.saving import save_artifacts
from ml.training import normal_loop, federated_loop, History
from ml.model_list import MODELS
from common.config import MODELS_DIR, DATASETS_DIR, SEED
from ..common.plots import plot_history
from ..common.reports import RUN_MANIFEST, get_report_dir, write_history_csv, write_yaml

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


def run_loop(trainer: Trainer, data_dir: Path, loop: str, eval_ids: list[str],
             steps: int, local_epochs: int
             ) -> tuple[History, tf.data.Dataset, list[str], list[str]]:
    datasets = trainer.subject_datasets(data_dir)
    sids = [d.name for d in subject_dirs(data_dir, trainer.data_subdir)]
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
    parser.add_argument('--loop', choices=LOOP_OPTIONS, default='normal', help='Training loop (default: normal)')
    parser.add_argument('--eval-subjects', default='14-15',
                        help="Subjects held out whole for evaluation (default: 14-15). Either "
                             "a single id N (subject SN, LOSO-style), an inclusive id range "
                             "'n-m' (Sn..Sm), a comma-separated id list 'i,j,k', or 'none' to "
                             "train on every subject and skip evaluation (all-users teacher). "
                             "Both loops train on the rest and score on the held-out set.")
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for the normal loop')
    parser.add_argument('--local-epochs', type=int, default=2, help='Local epochs per round (federated)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the model default batch size. Artifacts from a '
                             'non-default batch size are suffixed (e.g. trainable_32.tflite).')
    parser.add_argument('--dataset-dir', type=Path, default=DATASETS_DIR,
                        help='Dataset directory to train on (default: datasets). Point this '
                             'at an alternative source with the same structure as datasets/ '
                             'to train against alternative labels (e.g. a teacher\'s distilled '
                             'ones) instead of the synthetic ground truth.')
    args = parser.parse_args()

    data_dir = args.dataset_dir

    result_dir = MODELS_DIR / args.model
    result_dir.mkdir(parents=True, exist_ok=True)

    report_dir = get_report_dir(args.model, args.loop)

    trainer = MODELS[args.model].build_trainer(data_dir, batch_size=args.batch_size)
    sids = [d.name for d in subject_dirs(data_dir, trainer.data_subdir)]
    eval_ids = parse_eval_selection(args.eval_subjects, sids)
    history, eval_dataset, train_ids, held_ids = run_loop(
        trainer, data_dir, args.loop, eval_ids, args.epochs, args.local_epochs)

    batch_size = trainer.model.batch_size
    postfix = (f'_{batch_size}'
               if batch_size != type(trainer.model).default_batch_size else '')
    save_artifacts(trainer, result_dir, eval_dataset, postfix, data_root=data_dir)

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
        'metric': trainer.primary_metric,
        'epochs': args.epochs,
        'step_unit': 'round' if args.loop == 'federated' else 'epoch',
        'local_epochs': args.local_epochs if args.loop == 'federated' else None,
        'clients': len(train_ids) if args.loop == 'federated' else None,
        'train_subjects': train_ids,
        'eval_subjects': held_ids,
        'batch_size': batch_size,
        'dataset_dir': args.dataset_dir,
        'seed': SEED,
        'history': 'training.csv',
        'final': {'loss': final_loss, **final_metrics},
    })
