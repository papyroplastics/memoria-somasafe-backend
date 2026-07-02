"""
Train a SomaSafe model with a chosen training loop and export
SavedModel + TFLite artifacts into results/<model>.
"""

import argparse
from pathlib import Path
import tensorflow as tf

from ml.models.common import Trainer
from ml.saving import save_artifacts
from ml.training import normal_loop, federated_loop, History
from ml.model_list import MODELS
from common.config import MODELS_DIR, DATASETS_DIR, SEED
from .common.post_train import plot_history, get_report_dir

LOOP_OPTIONS = ['normal', 'federated']

def run_loop(trainer: Trainer, data_dir: Path, loop: str, split: float,
             epochs: int, local_epochs: int) -> tuple[History, tf.data.Dataset]:

    if loop == 'normal':
        train_dataset, eval_dataset = trainer.combined_datasets(data_dir, split)
        history = normal_loop(trainer, train_dataset, eval_dataset, epochs)

    elif loop == 'federated':
        train_datasets, eval_dataset  = trainer.subject_datasets(data_dir, SEED)
        history = federated_loop(trainer, train_datasets, eval_dataset, local_epochs, epochs)

    else:
        raise Exception(f"Invalid loop type: {loop}")

    return history, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to train')
    parser.add_argument('--loop', choices=LOOP_OPTIONS, default='normal', help='Training loop (default: normal)')
    parser.add_argument('--split', type=float, default=0.9, help='Train/eval data split')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs for the normal loop')
    parser.add_argument('--local-epochs', type=int, default=5, help='Local epochs per round (federated)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the model default batch size. Artifacts from a '
                             'non-default batch size are suffixed (e.g. trainable_32.tflite).')
    parser.add_argument('--dataset-dir', type=Path, default=DATASETS_DIR,
                        help='Dataset directory to train on (default: datasets). Point this '
                             'at an alternative source with the same structure as datasets/ '
                             '(e.g. a distilled-labels directory) to train against distilled '
                             'labels instead of the synthetic ground truth.')
    args = parser.parse_args()

    data_dir = args.dataset_dir

    result_dir = MODELS_DIR / args.model
    report_dir = get_report_dir(result_dir)

    trainer = MODELS[args.model].build_trainer(batch_size=args.batch_size)
    history, eval_dataset = run_loop(trainer, data_dir, args.loop, args.split, args.epochs, args.local_epochs)

    postfix = '' if trainer.batch_size == trainer.default_batch_size else f'_{trainer.batch_size}'
    save_artifacts(trainer, result_dir, eval_dataset, data_dir, postfix)
    plot_history(history, trainer.primary_metric, report_dir)
    trainer.report(report_dir, eval_dataset)
