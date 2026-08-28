"""
Seed a default-batch model with the weights of one trained at a larger batch size and
fine-tune it with the normal loop. Artifacts go to shared/gen/models/<model>; the
training report to results/<model>.
"""

import argparse

from common.config import DATASETS_DIR, MODELS_DIR
from ml.loading import holdout, pool
from ml.training import normal_loop
from ml.saving import load_weights, save_artifacts, weights_path
from ml.model_list import MODELS
from ..common.plots import plot_history
from ..common.reports import get_report_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Model to transfer')
    parser.add_argument('source_batch_size', type=int,
                        help='Batch size of the already-trained source run, which must be '
                             'tagged with it')
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
    target_batch_size = target_trainer.model.batch_size
    if args.source_batch_size < target_batch_size:
        raise SystemExit(
            f"source batch size ({args.source_batch_size}) must be >= the default "
            f"batch size ({target_batch_size}) of '{args.model}'")

    # Source: rebuilt at its batch size, weights restored from its saved .npy.
    source_trainer = MODELS[args.model].build_trainer(data_dir, batch_size=args.source_batch_size)
    source_path = weights_path(result_dir, str(args.source_batch_size))
    if not source_path.exists():
        raise SystemExit(
            f"source weights not found at {source_path}. Train them first with "
            f"`train {args.model} --batch-size {args.source_batch_size} "
            f"--tag {args.source_batch_size}`.")
    source_trainer.model.restore(load_weights(source_path))

    target_trainer.model.transfer_from(source_trainer.model)
    print(f"Transferred weights from {source_path} into a batch-size "
          f"{target_batch_size} {args.model}")

    train_subjects, held_out = holdout(target_trainer.subject_datasets(),
                                       args.eval_subjects)
    eval_dataset = pool(held_out)
    history = normal_loop(target_trainer, pool(train_subjects), eval_dataset, args.epochs)

    save_artifacts(target_trainer, result_dir)
    plot_history(history, target_trainer.primary_metric, report_dir)
    target_trainer.report(report_dir, eval_dataset)
