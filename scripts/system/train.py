"""
Train a SomaSafe model with a chosen training loop. Serving artifacts
(trainable/quantized .tflite) go to shared/gen/models/<model>; the training
history, plot, run manifest and eval report go to results/<model>/<loop>.

Both loops hold out the last --eval-subjects subjects whole and score on them, so a
centralized and a federated run at the same --eval-subjects train on the same data and
are directly comparable — that overlay is what scripts.figures.plot_convergence draws
from the manifests this writes.
"""

import argparse
from pathlib import Path
import tensorflow as tf

from ml.loading import holdout, pool
from ml.models.common import Trainer
from ml.saving import save_artifacts
from ml.training import normal_loop, federated_loop, History
from ml.model_list import MODELS
from common.config import MODELS_DIR, DATASETS_DIR, SEED
from ..common.plots import plot_history
from ..common.reports import RUN_MANIFEST, get_report_dir, write_history_csv, write_yaml

LOOP_OPTIONS = ['normal', 'federated']


def run_loop(trainer: Trainer, data_dir: Path, loop: str, eval_subjects: int,
             steps: int, local_epochs: int) -> tuple[History, tf.data.Dataset, int]:
    train_subjects, held_out = holdout(trainer.subject_datasets(data_dir), eval_subjects)
    eval_dataset = pool(held_out)

    if loop == 'normal':
        history = normal_loop(trainer, pool(train_subjects), eval_dataset, steps)

    elif loop == 'federated':
        history = federated_loop(trainer, train_subjects, eval_dataset, local_epochs, steps)

    else:
        raise Exception(f"Invalid loop type: {loop}")

    return history, eval_dataset, len(train_subjects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to train')
    parser.add_argument('--loop', choices=LOOP_OPTIONS, default='normal', help='Training loop (default: normal)')
    parser.add_argument('--eval-subjects', type=int, default=2,
                        help='Subjects held out whole for evaluation (default: 2). The last '
                             'N subjects; both loops train on the rest and score on these.')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs for the normal loop')
    parser.add_argument('--rounds', type=int, default=5, help='Global rounds for the federated loop')
    parser.add_argument('--local-epochs', type=int, default=2, help='Local epochs per round (federated)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the model default batch size. Artifacts from a '
                             'non-default batch size are suffixed (e.g. trainable_32.tflite).')
    parser.add_argument('--dataset-dir', type=Path, default=DATASETS_DIR,
                        help='Dataset directory to train on (default: datasets). Point this '
                             'at an alternative source with the same structure as datasets/ '
                             '(e.g. a distilled-labels directory) to train against distilled '
                             'labels instead of the synthetic ground truth.')
    parser.add_argument('--tag', default=None,
                        help='Name this run so it does not overwrite the canonical one: '
                             'results go to results/<model>/<loop>-<tag>/ and artifacts are '
                             'suffixed (trainable_<tag>.tflite). Use it for a variant of a '
                             'model you also train normally — e.g. the same student trained '
                             'on distilled labels rather than the synthetic ground truth.')
    args = parser.parse_args()

    if args.eval_subjects < 1:
        raise SystemExit("--eval-subjects must be >= 1: the run scores on the held-out "
                         "subjects and the manifest records that metric.")

    data_dir = args.dataset_dir

    result_dir = MODELS_DIR / args.model
    result_dir.mkdir(parents=True, exist_ok=True)

    report_dir = get_report_dir(args.model, args.loop if args.tag is None
                                            else f'{args.loop}-{args.tag}')

    trainer = MODELS[args.model].build_trainer(data_dir, batch_size=args.batch_size)
    steps = args.rounds if args.loop == 'federated' else args.epochs
    history, eval_dataset, n_train_subjects = run_loop(
        trainer, data_dir, args.loop, args.eval_subjects, steps, args.local_epochs)

    # Both the tag and a non-default batch size keep a run's artifacts off the canonical
    # names, so a variant never clobbers the model the system actually serves.
    batch_size = trainer.model.batch_size
    parts = ([args.tag] if args.tag else []) + (
        [str(batch_size)] if batch_size != type(trainer.model).default_batch_size else [])
    postfix = ''.join(f'_{p}' for p in parts)
    save_artifacts(trainer, result_dir, eval_dataset, postfix)
    plot_history(history, trainer.primary_metric, report_dir)
    write_history_csv(history, report_dir)
    trainer.report(report_dir, eval_dataset)

    _, final_loss, final_metrics = history[-1]
    write_yaml(report_dir / RUN_MANIFEST, {
        'model': args.model,
        'loop': args.loop,
        'tag': args.tag,
        'metric': trainer.primary_metric,
        'steps': steps,
        'step_unit': 'round' if args.loop == 'federated' else 'epoch',
        'local_epochs': args.local_epochs if args.loop == 'federated' else None,
        'clients': n_train_subjects if args.loop == 'federated' else None,
        'train_subjects': n_train_subjects,
        'eval_subjects': args.eval_subjects,
        'batch_size': batch_size,
        'dataset_dir': args.dataset_dir,
        'seed': SEED,
        'history': 'training.csv',
        'final': {'loss': final_loss, **final_metrics},
    })
