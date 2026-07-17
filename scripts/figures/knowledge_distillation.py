"""
Knowledge-distillation round-trip + personalization probe (report Secs. 5.4/5.8), on a
teacher trained on ALL users. The autoencoder teacher's soft labels train a supervised
FeatureMLP student that a wearable can run; this script distils those labels in memory and
measures whether personalizing the student on a user's own labels beats the shared model —
for both the float and int8 (quantized) deployment.

Personalization is **leave-one-subject-out**: for each subject, a fresh global student is
trained (centralized, in-script) on the *other* subjects' distilled labels, then fine-tuned
on the held-out subject's own distilled labels over a chronological train split, and both
models are scored — float and int8 — on that subject's held-out split against the **true**
labels (never the distilled ones). Rotating the held-out subject keeps every fold
leakage-free (the global never trained on the subject it is judged on) and makes none
special. The teacher trained on all users so every subject's distilled labels are the same
(teacher-seen) quality, so the folds are comparable.

The expected FPR is calibrated inline (cheap) on all subjects; the labels are the sigmoid of
each window's reconstruction error past the subject's own threshold, scaled by its own
clean-error std. Nothing is written to disk but the final metrics.

    uv run -m scripts.figures.knowledge_distillation cnn-ae --weights <all-users teacher>
"""


import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common.config import MODELS_DIR, DATASETS_DIR
from ml.preprocessing import MIXED_FEATURE_SUBDIR, get_sorted_paths
from ml.metrics import classification_report
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights, get_optimized_model
from ..common.litert import infer_int8
from ..common.reports import get_report_dir, write_metrics_csv
from ..common.scoring import (
    DETECTOR, calibrate_expected_fpr, clean_threshold, score_dir_by_subject,
    score_mixed_by_subject, load_mixed_truth,
)

VARIANTS = ('global_float', 'global_int8', 'personal_float', 'personal_int8')


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def distilled_labels(mixed: dict[str, dict[str, np.ndarray]],
                     clean: dict[str, dict[str, np.ndarray]], expected_fpr: float
                     ) -> dict[str, np.ndarray]:
    """Each subject's soft [0,1] labels: sigmoid of the error's signed distance to the
    subject's own threshold, scaled by its own clean-error std (label > 0.5 == the hard
    decision)."""
    labels = {}
    for sid in mixed:
        thr = clean_threshold(clean[sid][DETECTOR], expected_fpr)
        scale = float(clean[sid][DETECTOR].std()) + 1e-8
        labels[sid] = sigmoid((mixed[sid][DETECTOR] - thr) / scale)
    return labels


def train_on(model, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.reshape(-1, 1).astype(np.float32))
    ).batch(batch_size, drop_remainder=True)
    for _ in range(epochs):
        for xb, yb in ds:
            model.train(xb, yb)


def eval_logits_float(model, X: np.ndarray) -> np.ndarray:
    """Per-window logits from the float model (eval z-scores raw features)."""
    out = np.empty(len(X), dtype=np.float32)
    for i, x in enumerate(X):
        logits = model.eval(tf.constant(x.reshape(1, -1), dtype=tf.float32))['logits']
        out[i] = float(np.asarray(logits).reshape(-1)[0])
    return out


def load_features(data_dir: Path, sid: str) -> np.ndarray:
    return np.load(data_dir / MIXED_FEATURE_SUBDIR / sid / 'features.npy').astype(np.float32)


def load_true(data_dir: Path, sid: str) -> np.ndarray:
    return np.load(data_dir / MIXED_FEATURE_SUBDIR / sid / 'labels.npy').reshape(-1) > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('teacher', choices=sorted(MODELS),
                        help='Autoencoder trained on ALL users, whose soft labels train the student')
    parser.add_argument('--student', default='feature-mlp', choices=sorted(MODELS),
                        help='Student model to distil into + personalize (default: feature-mlp)')
    parser.add_argument('--weights', type=Path, default=None,
                        help='Teacher all-users trainable .tflite '
                             '(default: shared/gen/models/<teacher>/trainable.tflite)')
    parser.add_argument('--global-epochs', type=int, default=5,
                        help="Epochs to train each fold's global student (default: 5)")
    parser.add_argument('--epochs', type=int, default=5,
                        help='Fine-tune (personalization) epochs (default: 5)')
    parser.add_argument('--train-split', type=float, default=0.6,
                        help='Fraction of the held-out subject used to fine-tune; the rest '
                             'is the eval split (default: 0.6)')
    parser.add_argument('--global-batch-size', type=int, default=32,
                        help='Batch size for global student training (default: 32)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Fine-tune/eval batch size (default: 1, as on-device)')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    weights_path = args.weights or (MODELS_DIR / args.teacher / 'trainable.tflite')
    if not weights_path.exists():
        raise SystemExit(f"teacher weights not found at {weights_path}.")

    teacher = MODELS[args.teacher].build_trainer(data_dir)
    teacher.model.restore(load_trainable_weights(weights_path))
    assert isinstance(teacher, AutoencoderTrainer)

    print("Scoring the teacher over all subjects + calibrating the expected FPR...")
    mixed = score_mixed_by_subject(teacher, data_dir)
    truth = load_mixed_truth(data_dir, mixed)
    clean = score_dir_by_subject(teacher, data_dir, None)
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; cannot threshold.")
    expected_fpr, _ = calibrate_expected_fpr(clean, mixed, truth)
    distilled = distilled_labels(mixed, clean, expected_fpr)
    print(f"expected_fpr={expected_fpr:.4f}; distilled soft labels for {len(distilled)} subjects")

    # Population calibration set for int8 export — shared by every model so the int8
    # comparison isolates the weights, not the calibration feed.
    base = MODELS[args.student].build_trainer(data_dir, args.batch_size)
    rep_dataset = base.representative_dataset(data_root=data_dir)
    feat_mean = base.model.feat_mean.numpy()
    feat_std = base.model.feat_std.numpy()

    subjects = [d.name for d in get_sorted_paths(data_dir / MIXED_FEATURE_SUBDIR)]
    print(f"\nLeave-one-subject-out personalization over {len(subjects)} subjects "
          f"(global_epochs={args.global_epochs}, epochs={args.epochs}, "
          f"train_split={args.train_split}):\n")

    pooled = {v: {'pred': [], 'truth': []} for v in VARIANTS}
    rows = []
    print(f"  {'held-out':<9} " + "  ".join(f"{v:>14}" for v in VARIANTS) + "   (F1)")
    for sid in subjects:
        # Global student: trained on every *other* subject's distilled labels.
        Xs, ys = [], []
        for s in subjects:
            if s == sid:
                continue
            Xo, yo = load_features(data_dir, s), distilled[s]
            n = min(len(Xo), len(yo))
            Xs.append(Xo[:n]); ys.append(yo[:n])
        X_global, y_global = np.concatenate(Xs), np.concatenate(ys)

        gtrainer = MODELS[args.student].build_trainer(data_dir, args.global_batch_size)
        train_on(gtrainer.model, X_global, y_global, args.global_epochs, args.global_batch_size)
        global_weights = np.asarray(gtrainer.model.save()['weights'])

        # Held-out subject: chronological split; fine-tune on its own distilled labels,
        # evaluate against the true ones.
        X, y_true = load_features(data_dir, sid), load_true(data_dir, sid)
        y_distill = distilled[sid]
        n = min(len(X), len(y_true), len(y_distill))
        X, y_true, y_distill = X[:n], y_true[:n], y_distill[:n]
        n_train = int(n * args.train_split)
        X_ev, y_ev = X[n_train:], y_true[n_train:]
        X_ev_norm = (X_ev - feat_mean) / feat_std

        # Global + personal, both at the on-device batch size, restored from the trained
        # global weights (batch-independent for an MLP).
        global_model = MODELS[args.student].build_trainer(data_dir, args.batch_size).model
        global_model.restore(tf.constant(global_weights, dtype=tf.float32))
        global_int8 = get_optimized_model(global_model, rep_dataset)

        ptrainer = MODELS[args.student].build_trainer(data_dir, args.batch_size)
        ptrainer.model.restore(tf.constant(global_weights, dtype=tf.float32))
        train_on(ptrainer.model, X[:n_train], y_distill[:n_train], args.epochs, args.batch_size)
        personal_int8 = get_optimized_model(ptrainer.model, rep_dataset)

        logits = {
            'global_float': eval_logits_float(global_model, X_ev),
            'global_int8': infer_int8(global_int8, X_ev_norm),
            'personal_float': eval_logits_float(ptrainer.model, X_ev),
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
        print(f"  {sid:<9} " + "  ".join(f"{row[f'{v}_f1']:>14.3f}" for v in VARIANTS))

    print("\npooled over all eval windows:")
    print(f"  {'variant':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'accuracy':>10}")
    overall = {}
    for v in VARIANTS:
        rep = classification_report(np.concatenate(pooled[v]['pred']),
                                    np.concatenate(pooled[v]['truth']))
        overall[v] = rep
        print(f"  {v:<16} {rep['precision']:>10.4f} {rep['recall']:>10.4f} "
              f"{rep['f1']:>10.4f} {rep['accuracy']:>10.4f}")

    print(f"\npersonalization Δf1 (personal − global):  "
          f"float={overall['personal_float']['f1'] - overall['global_float']['f1']:+.4f}  "
          f"int8={overall['personal_int8']['f1'] - overall['global_int8']['f1']:+.4f}")

    report_dir = get_report_dir(args.student, 'personalization')
    write_metrics_csv(rows, report_dir, 'personalization.csv')
    (report_dir / 'personalization.json').write_text(json.dumps({
        'teacher': args.teacher, 'student': args.student, 'expected_fpr': expected_fpr,
        'global_epochs': args.global_epochs, 'epochs': args.epochs,
        'train_split': args.train_split, 'batch_size': args.batch_size,
        'holdout': 'leave-one-subject-out', 'overall': overall, 'per_subject': rows,
    }, indent=2))
    print(f"wrote report to {report_dir}/")
