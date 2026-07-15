"""
Test whether personalizing the FeatureMLP on a single user's own distilled labels beats
the shared global model — for both the float and the int8 (quantized) deployment.

For each subject: start from the global weights (trained on every subject's distilled
labels), fine-tune on that subject's *distilled* labels over a chronological train split,
quantize, and score the global and fine-tuned models — float and int8 — on the held-out
eval split against the **true** labels (never the distilled ones). The train/eval split is
contiguous in time (past -> future), so eval windows are never adjacent to the ones
fine-tuning saw. Both models are quantized with the same population calibration set (as
the server's quantize path does), so the int8 comparison isolates the weights.

    uv run -m scripts.personalize_test --teacher cnn-ae
"""


import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common.config import MODELS_DIR, DATASETS_DIR, RESULTS_DIR
from ml.preprocessing import MIXED_FEATURE_SUBDIR, get_sorted_paths
from ml.metrics import classification_report
from ml.model_list import MODELS
from ml.saving import load_trainable_weights, get_optimized_model
from ..common.litert import infer_int8
from ..common.reports import get_report_dir, write_metrics_csv

VARIANTS = ('global_float', 'global_int8', 'personal_float', 'personal_int8')


def eval_logits_float(model, X: np.ndarray) -> np.ndarray:
    """Per-window logits from the float model (batch_size 1; eval z-scores raw features)."""
    out = np.empty(len(X), dtype=np.float32)
    for i, x in enumerate(X):
        logits = model.eval(tf.constant(x.reshape(1, -1), dtype=tf.float32))['logits']
        out[i] = float(np.asarray(logits).reshape(-1)[0])
    return out


def fine_tune(model, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.astype(np.float32))).batch(batch_size, drop_remainder=True)
    for _ in range(epochs):
        for xb, yb in ds:
            model.train(xb, yb)


def load_subject(data_dir: Path, distilled_dir: Path, sid: str):
    """Raw features, distilled (teacher) labels and true labels for one subject, aligned
    to a common window count."""
    X = np.load(data_dir / MIXED_FEATURE_SUBDIR / sid / 'features.npy').astype(np.float32)
    y_distill = np.load(distilled_dir / MIXED_FEATURE_SUBDIR / sid / 'labels.npy').reshape(-1, 1)
    y_true = np.load(data_dir / MIXED_FEATURE_SUBDIR / sid / 'labels.npy').reshape(-1, 1)
    n = min(len(X), len(y_distill), len(y_true))
    return X[:n], y_distill[:n].astype(np.float32), (y_true[:n].reshape(-1) > 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', default='feature-mlp', choices=sorted(MODELS),
                        help='Student model to personalize (default: feature-mlp)')
    parser.add_argument('--teacher', default='cnn-ae',
                        help='Autoencoder whose distilled-labels tree to fine-tune on '
                             '(results/<teacher>/distilled-labels; default: cnn-ae)')
    parser.add_argument('--weights', type=Path, default=None,
                        help='Global trainable .tflite (default: shared/gen/models/<model>/trainable.tflite)')
    parser.add_argument('--epochs', type=int, default=5, help='Fine-tune epochs (default: 5)')
    parser.add_argument('--train-split', type=float, default=0.6,
                        help='Fraction of each subject used to fine-tune; the rest is the '
                             'held-out eval split (default: 0.6)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fine-tune/eval batch size (default: 1, as on-device)')
    parser.add_argument('--eval-subject', default=None,
                        help='Only fine-tune + evaluate this one subject (e.g. S15) '
                             'instead of all. Pair it with a global model trained with '
                             'this subject held out for a leakage-free personalization '
                             'estimate.')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    result_dir = MODELS_DIR / args.model
    distilled_dir = RESULTS_DIR / args.teacher / 'distilled-labels'
    weights_path = args.weights or (result_dir / 'trainable.tflite')

    if not weights_path.exists():
        raise SystemExit(f"global weights not found at {weights_path}. Train '{args.model}' "
                         f"on the distilled labels first.")
    if not (distilled_dir / MIXED_FEATURE_SUBDIR).exists():
        raise SystemExit(f"no distilled labels at {distilled_dir}. Run distill_labels "
                         f"'{args.teacher}' first.")

    global_weights = load_trainable_weights(weights_path)

    # Population calibration set for int8 export — the same feed the server's quantize
    # path uses, shared by both models so the int8 comparison isolates the weights.
    base_trainer = MODELS[args.model].build_trainer(data_dir, args.batch_size)
    base_trainer.model.restore(tf.constant(global_weights, dtype=tf.float32))
    rep_dataset = base_trainer.representative_dataset(data_root=data_dir)
    feat_mean = base_trainer.model.feat_mean.numpy()
    feat_std = base_trainer.model.feat_std.numpy()

    print("Quantizing the global model...")
    global_int8 = get_optimized_model(base_trainer.model, rep_dataset)

    subjects = [d.name for d in get_sorted_paths(data_dir / MIXED_FEATURE_SUBDIR)]
    if args.eval_subject is not None:
        if args.eval_subject not in subjects:
            raise SystemExit(f"--eval-subject {args.eval_subject!r} not among {subjects}")
        subjects = [args.eval_subject]
    print(f"Personalizing {args.model} from {args.teacher} labels over {len(subjects)} "
          f"subjects (epochs={args.epochs}, train_split={args.train_split}, "
          f"batch_size={args.batch_size}):\n")

    pooled = {v: {'pred': [], 'truth': []} for v in VARIANTS}
    rows = []
    header = f"  {'subject':<8} " + "  ".join(f"{v:>14}" for v in VARIANTS) + "   (F1)"
    print(header)
    for sid in subjects:
        X, y_distill, y_true = load_subject(data_dir, distilled_dir, sid)
        n_train = int(len(X) * args.train_split)
        X_tr, y_tr = X[:n_train], y_distill[:n_train]
        X_ev, y_ev = X[n_train:], y_true[n_train:]
        X_ev_norm = (X_ev - feat_mean) / feat_std

        # Fresh model + optimizer per subject, seeded from the global weights.
        trainer = MODELS[args.model].build_trainer(data_dir, args.batch_size)
        trainer.model.restore(tf.constant(global_weights, dtype=tf.float32))
        fine_tune(trainer.model, X_tr, y_tr, args.epochs, args.batch_size)
        personal_int8 = get_optimized_model(trainer.model, rep_dataset)

        logits = {
            'global_float': eval_logits_float(base_trainer.model, X_ev),
            'global_int8': infer_int8(global_int8, X_ev_norm),
            'personal_float': eval_logits_float(trainer.model, X_ev),
            'personal_int8': infer_int8(personal_int8, X_ev_norm),
        }
        row = {'subject': sid, 'n_eval': len(X_ev)}
        for v in VARIANTS:
            pred = logits[v] > 0.0
            rep = classification_report(pred, y_ev)
            pooled[v]['pred'].append(pred)
            pooled[v]['truth'].append(y_ev)
            for m in ('precision', 'recall', 'f1', 'accuracy'):
                row[f'{v}_{m}'] = rep[m]
        rows.append(row)
        print(f"  {sid:<8} " + "  ".join(f"{row[f'{v}_f1']:>14.3f}" for v in VARIANTS))

    print("\npooled over all eval windows:")
    print(f"  {'variant':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'accuracy':>10}")
    overall = {}
    for v in VARIANTS:
        pred = np.concatenate(pooled[v]['pred'])
        truth = np.concatenate(pooled[v]['truth'])
        rep = classification_report(pred, truth)
        overall[v] = rep
        print(f"  {v:<16} {rep['precision']:>10.4f} {rep['recall']:>10.4f} "
              f"{rep['f1']:>10.4f} {rep['accuracy']:>10.4f}")

    print(f"\npersonalization Δf1 (personal − global):  "
          f"float={overall['personal_float']['f1'] - overall['global_float']['f1']:+.4f}  "
          f"int8={overall['personal_int8']['f1'] - overall['global_int8']['f1']:+.4f}")

    report_dir = get_report_dir(args.model, 'personalization')
    suffix = f'_{args.eval_subject}' if args.eval_subject else ''
    write_metrics_csv(rows, report_dir, f'personalization{suffix}.csv')
    (report_dir / f'personalization{suffix}.json').write_text(json.dumps({
        'model': args.model, 'teacher': args.teacher, 'epochs': args.epochs,
        'train_split': args.train_split, 'batch_size': args.batch_size,
        'overall': overall, 'per_subject': rows,
    }, indent=2))
    print(f"wrote report to {report_dir}/")
