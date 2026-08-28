"""Distillation + leave-one-subject-out personalization (report Secs. 5.4/5.8) on a teacher
trained on ALL users: per fold a fresh student is trained on the *other* subjects' soft
labels, fine-tuned on the held-out subject's own, and both are scored (float + int8) against
that subject's TRUE labels, alongside a `direct_float` ceiling trained on true labels instead
of the teacher's. Only the final metrics are written to disk."""


import argparse

import numpy as np
import tensorflow as tf

from common.config import MODELS_DIR, DATASETS_DIR
from ml.dataset_list import DATASETS
from ml.metrics import classification_report
from ml.model_list import MODELS
from ml.saving import load_weights, get_optimized_model, weights_path
from ml.sources.common import DataSource
from ml.sources.dalia import CLEAN, MIXED
from ..common.litert import infer_int8
from ..common.reports import get_report_dir, write_metrics_csv, write_yaml
from ..common.scoring import (
    calibrate_expected_fpr, clean_threshold, eval_padded, score_subjects, mixed_truth,
)

VARIANTS = ('global_float', 'global_int8', 'personal_float', 'personal_int8', 'direct_float')


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def distilled_labels(mixed: dict[str, np.ndarray], clean: dict[str, np.ndarray],
                     expected_fpr: float) -> dict[str, np.ndarray]:
    labels = {}
    for sid in mixed:
        thr = clean_threshold(clean[sid], expected_fpr)
        scale = float(clean[sid].std()) + 1e-8
        labels[sid] = sigmoid((mixed[sid] - thr) / scale)
    return labels


def train_on(model, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
    ds = tf.data.Dataset.from_tensor_slices(
        (X.astype(np.float32), y.reshape(-1, 1).astype(np.float32))
    ).batch(batch_size, drop_remainder=True)
    for _ in range(epochs):
        for xb, yb in ds:
            model.train(xb, yb)


def eval_logits_float(model, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)
    out = eval_padded(model, X.astype(np.float32))
    return out['logits'].reshape(-1).astype(np.float32)


def load_features(source: DataSource, sid: str) -> np.ndarray:
    return source.features(sid)[0]



def load_true(source: DataSource, sid: str) -> np.ndarray:
    return source.window_labels(sid) > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('teacher', choices=sorted(MODELS),
                        help='Autoencoder trained on ALL users, whose soft labels train the student')
    parser.add_argument('--student', default='feature-mlp', choices=sorted(MODELS),
                        help='Student model to distil into + personalize')
    parser.add_argument('--tag', default=None,
                        help="Tag of the teacher's all-users train.py run")
    parser.add_argument('--global-epochs', type=int, default=5,
                        help="Epochs to train each fold's global student")
    parser.add_argument('--epochs', type=int, default=5,
                        help='Fine-tune (personalization) epochs')
    parser.add_argument('--train-split', type=float, default=0.6,
                        help='Fraction of the held-out subject used to fine-tune; the rest '
                             'is the eval split')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training, fine-tuning and evaluation')
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='ppg-dalia',
                        help='Dataset the teacher scores and the student trains on; '
                             'ppg-dalia-low keeps only the low-activity windows')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    source = DATASETS[args.dataset].build(data_dir)
    teacher_weights = weights_path(MODELS_DIR / args.teacher, args.tag)
    if not teacher_weights.exists():
        raise SystemExit(f"teacher weights not found at {teacher_weights}.")

    teacher = MODELS[args.teacher].build_model(data_dir)
    teacher.restore(load_weights(teacher_weights))

    print(f"Scoring the teacher over all subjects of {DATASETS[args.dataset].name} "
          f"+ calibrating the expected FPR...")
    truth = mixed_truth(source)
    clean = score_subjects(teacher, source, CLEAN)
    mixed = score_subjects(teacher, source, MIXED)

    expected_fpr = calibrate_expected_fpr(clean, mixed, truth)
    distilled = distilled_labels(mixed, clean, expected_fpr)
    print(f"expected_fpr={expected_fpr:.4f}; distilled soft labels for {len(distilled)} subjects")

    # Population calibration set for int8 export — shared by every model so the int8
    # comparison isolates the weights, not the calibration feed. The only thing a trainer
    # is needed for here: it owns the feed dict the int8 converter expects.
    base = MODELS[args.student].build_trainer(data_dir, args.batch_size)
    rep_dataset = base.representative_dataset()
    feat_mean = base.model.feat_mean.numpy()
    feat_std = base.model.feat_std.numpy()

    subjects = source.subject_ids()
    print(f"\nLeave-one-subject-out personalization over {len(subjects)} subjects "
          f"(global_epochs={args.global_epochs}, epochs={args.epochs}, "
          f"train_split={args.train_split}):\n")

    pooled = {v: {'pred': [], 'truth': []} for v in VARIANTS}
    rows = []
    print(f"  {'held-out':<9} " + "  ".join(f"{v:>14}" for v in VARIANTS) + "   (F1)")
    for sid in subjects:
        # Global student: trained on every *other* subject's distilled labels. The direct
        # student sees the same windows with their true labels instead, so the two differ
        # only in where the supervision came from.
        Xs, ys, ys_true = [], [], []
        for s in subjects:
            if s == sid:
                continue
            Xs.append(load_features(source, s))
            ys.append(distilled[s])
            ys_true.append(load_true(source, s))
        X_global = np.concatenate(Xs)
        y_global, y_global_true = np.concatenate(ys), np.concatenate(ys_true)

        gmodel = MODELS[args.student].build_model(data_dir, args.batch_size)
        train_on(gmodel, X_global, y_global, args.global_epochs, args.batch_size)
        global_weights = np.asarray(gmodel.save()['weights'])

        dmodel = MODELS[args.student].build_model(data_dir, args.batch_size)
        train_on(dmodel, X_global, y_global_true, args.global_epochs, args.batch_size)
        direct_weights = np.asarray(dmodel.save()['weights'])

        # Held-out subject: chronological split; fine-tune on its own distilled labels,
        # evaluate against the true ones.
        X, y_true = load_features(source, sid), load_true(source, sid)
        y_distill = distilled[sid]
        n_train = int(len(X) * args.train_split)
        X_ev, y_ev = X[n_train:], y_true[n_train:]
        X_ev_norm = (X_ev - feat_mean) / feat_std

        global_model = MODELS[args.student].build_model(data_dir, args.batch_size)
        global_model.restore(tf.constant(global_weights, dtype=tf.float32))
        global_int8 = get_optimized_model(global_model, rep_dataset)

        personal_model = MODELS[args.student].build_model(data_dir, args.batch_size)
        personal_model.restore(tf.constant(global_weights, dtype=tf.float32))
        train_on(personal_model, X[:n_train], y_distill[:n_train], args.epochs, args.batch_size)
        personal_int8 = get_optimized_model(personal_model, rep_dataset)

        direct_model = MODELS[args.student].build_model(data_dir, args.batch_size)
        direct_model.restore(tf.constant(direct_weights, dtype=tf.float32))

        logits = {
            'global_float': eval_logits_float(global_model, X_ev),
            'global_int8': infer_int8(global_int8, X_ev_norm),
            'personal_float': eval_logits_float(personal_model, X_ev),
            'personal_int8': infer_int8(personal_int8, X_ev_norm),
            'direct_float': eval_logits_float(direct_model, X_ev),
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
    print(f"distillation cost Δf1 (direct − global):  "
          f"float={overall['direct_float']['f1'] - overall['global_float']['f1']:+.4f}")

    report_dir = get_report_dir(args.student, f'personalization/{args.dataset}')
    write_metrics_csv(rows, report_dir, 'personalization.csv')
    write_yaml(report_dir / 'personalization.yaml', {
        'dataset': {'key': args.dataset, 'name': DATASETS[args.dataset].name},
        'shows': f"Leave-one-subject-out personalization of a distilled {args.student} "
                 f"student against a {args.teacher} teacher (report Secs. 5.4/5.8): per-"
                 f"fold precision/recall/F1/accuracy for the global vs. personalized "
                 f"student, float and int8, plus a direct-supervision ceiling, all scored "
                 f"against each held-out subject's true labels.",
        'measured_on': {
            'holdout': 'leave-one-subject-out',
            'subjects': subjects,
            'note': "each fold's global student never trains on the subject it is judged "
                    "on; the teacher trained on all subjects so every fold's distilled "
                    "labels are the same, teacher-seen quality.",
        },
        'variants': {
            'global_float': "student trained on the other subjects' distilled soft labels",
            'global_int8': 'the same student quantized to int8',
            'personal_float': "global student fine-tuned on the held-out subject's own "
                              'distilled labels',
            'personal_int8': 'the same personalized student quantized to int8',
            'direct_float': "ceiling: same student, same windows, trained on the other "
                            "subjects' TRUE labels instead of the teacher's soft ones",
        },
        'config': {
            'teacher': args.teacher, 'student': args.student, 'expected_fpr': expected_fpr,
            'global_epochs': args.global_epochs, 'epochs': args.epochs,
            'train_split': args.train_split, 'batch_size': args.batch_size,
            'dataset': args.dataset,
        },
        'headline': overall,
        'personalization_delta_f1': {
            'float': overall['personal_float']['f1'] - overall['global_float']['f1'],
            'int8': overall['personal_int8']['f1'] - overall['global_int8']['f1'],
        },
        'distillation_cost_f1': {
            'float': overall['direct_float']['f1'] - overall['global_float']['f1'],
            'means': 'direct - global: what the student loses by learning from the '
                     'teacher instead of ground truth. Near zero means distillation '
                     'reproduces direct supervision; large and positive means the '
                     "teacher's labels are the bottleneck.",
        },
        'per_subject': 'see personalization.csv',
        'source': {'reproducible': True},
    })
    print(f"wrote report to {report_dir}/")
