"""
Transfer-learn a default-batch model from a model trained at a
larger batch size: copy the compatible trainable weights over,
then fine-tune with the normal loop. Serving artifacts go to
shared/gen/models/<model>; the training report to results/<model>.
"""

import argparse
from pathlib import Path
import tensorflow as tf

from common.config import DATASETS_DIR, MODELS_DIR, SEED
from ml.training import normal_loop
from ml.saving import load_trainable_weights, save_artifacts
from ml.model_list import MODELS
from ..common.post_train import plot_history, get_report_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to transfer')
    parser.add_argument('source_batch_size', type=int,
                        help='Batch size of the already-trained source artifact '
                             '(shared/gen/models/<model>/trainable_<N>.tflite)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Fine-tuning epochs after the weight transfer')
    parser.add_argument('--eval-subjects', type=int, default=2,
                        help='Subjects held out whole for evaluation')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    result_dir = MODELS_DIR / args.model
    report_dir = get_report_dir(args.model)

    # Target: a fresh model at the default batch size — the one we fine-tune and export.
    target_trainer = MODELS[args.model].build_trainer(data_dir)
    if args.source_batch_size < target_trainer.batch_size:
        raise SystemExit(
            f"source batch size ({args.source_batch_size}) must be >= the default "
            f"batch size ({target_trainer.batch_size}) of '{args.model}'")

    # Source: rebuilt at its batch size, weights restored from its saved trainable .tflite.
    source_trainer = MODELS[args.model].build_trainer(data_dir, batch_size=args.source_batch_size)
    source_path = result_dir / f'trainable_{args.source_batch_size}.tflite'
    if not source_path.exists():
        raise SystemExit(
            f"source artifact not found at {source_path}. Train it first with "
            f"`train {args.model} --batch-size {args.source_batch_size}`.")
    source_trainer.model.restore(tf.constant(load_trainable_weights(source_path)))

    target_trainer.model.transfer_from(source_trainer.model)
    print(f"Transferred weights from {source_path} into a batch-size "
          f"{target_trainer.batch_size} {args.model}")

    train_dataset, eval_dataset = target_trainer.combined_datasets(data_dir, args.eval_subjects)
    history = normal_loop(target_trainer, train_dataset, eval_dataset, args.epochs)

    save_artifacts(target_trainer, result_dir, eval_dataset)
    plot_history(history, target_trainer.primary_metric, report_dir)
    target_trainer.report(report_dir, eval_dataset)
