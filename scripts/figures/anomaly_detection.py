"""
Evaluate the autoencoder anomaly detector against the synthetic ground truth (report
Sec. 5.4). Takes a model trained **with a split**: it picks the expected FPR on the training
subjects (the model's own labeled population) and scores the detector on the **held-out**
subjects, so the numbers are generalization to an unseen user, consistent with the
convergence figures. Calibration happens inline — it is cheap, and this way the script is
self-contained (calibrate_fpr.py only exists to plot the whole sweep + ROC for the report).

Scores the detector against the true mixed-window labels and the per-type anomalous-signals/
sets: precision/recall/F1, per-anomaly-kind recall, and the empirical clean false-positive
rate. Writes the metrics to results/<model>/.
"""


import argparse
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.preprocessing import ANOMALOUS_SUBDIR, ANOMALY_KINDS, CLEAN_SUBDIR, MIXED_SUBDIR
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer
from ml.metrics import classification_report
from ml.saving import load_trainable_weights
from ..common.reports import get_report_dir, read_subject_split, write_yaml
from ..common.scoring import (
    DETECTOR, calibrate_expected_fpr, subject_thresholds, pooled_flags,
    score_dir_by_subject, load_mixed_truth,
)

EVAL_REPORT = 'anomaly_detection.yaml'   # detector metrics, from this script


def evaluate(trainer, data_dir: Path, clean: dict[str, dict[str, np.ndarray]],
             mixed: dict[str, dict[str, np.ndarray]], truth: dict[str, np.ndarray],
             thresholds: dict[str, dict[str, float]],
             subjects: set[str] | None = None) -> dict:
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])

    rep = classification_report(pooled_flags(mixed, thresholds), pooled_truth)
    detector = {
        'precision': rep['precision'], 'recall': rep['recall'], 'f1': rep['f1'],
        'accuracy': rep['accuracy'],
        'clean_fpr': float(pooled_flags(clean, thresholds).mean()),
    }

    anomalous_dir = data_dir / ANOMALOUS_SUBDIR
    per_kind = {}
    for name in ANOMALY_KINDS:
        sc = score_dir_by_subject(trainer, anomalous_dir / name, subjects=subjects)
        c = sum(len(v[DETECTOR]) for v in sc.values())
        per_kind[name] = {
            'count': c,
            'recall': float(pooled_flags(sc, thresholds).mean()) if c else None,
        }

    return {
        'n_windows': int(len(pooled_truth)),
        'gt_anomaly_rate': float(pooled_truth.mean()),
        'pred_anomaly_rate': float(pooled_flags(mixed, thresholds).mean()),
        'detector': detector,
        'per_kind': per_kind,
    }


def print_metrics(results: dict, expected_fpr: float):
    d = results['detector']
    print(f"\nexpected_fpr={expected_fpr:.4f}")
    print(f"detector ({DETECTOR}): accuracy={d['accuracy']:.4f} precision={d['precision']:.4f} "
          f"recall={d['recall']:.4f} f1={d['f1']:.4f} clean_fpr={d['clean_fpr']:.4f}")
    print(f"ground-truth anomaly rate={results['gt_anomaly_rate']:.1%}  "
          f"predicted rate={results['pred_anomaly_rate']:.1%}")

    print("\nrecall by anomaly kind (scored on per-type anomalous-signals/):")
    for name, stats in results['per_kind'].items():
        r = '     n/a' if stats['recall'] is None else f"{stats['recall']:>9.4f}"
        print(f"  {name:<9} {r}  ({stats['count']} windows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to evaluate')
    args = parser.parse_args()

    data_dir = DATASETS_DIR

    trainer = MODELS[args.model].build_trainer(data_dir)
    trainer.model.restore(load_trainable_weights(MODELS_DIR / args.model / 'trainable.tflite'))
    assert isinstance(trainer, AutoencoderTrainer)

    train_ids, held_out = read_subject_split(args.model, ('normal', 'federated'))
    train, held = set(train_ids), set(held_out)

    print(f"Calibrating expected FPR on the {len(train_ids)} training subjects...")
    mixed_tr = score_dir_by_subject(trainer, data_dir / MIXED_SUBDIR, subjects=train)
    truth_tr = load_mixed_truth(data_dir, mixed_tr)
    clean_tr = score_dir_by_subject(trainer, data_dir / CLEAN_SUBDIR, subjects=train)
    expected_fpr = calibrate_expected_fpr(clean_tr, mixed_tr, truth_tr)

    print(f"Evaluating on the {len(held_out)} held-out subjects: {', '.join(held_out)}")
    mixed = score_dir_by_subject(trainer, data_dir / MIXED_SUBDIR, subjects=held)
    truth = load_mixed_truth(data_dir, mixed)
    clean = score_dir_by_subject(trainer, data_dir / CLEAN_SUBDIR, subjects=held)
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; "
                         "cannot derive per-subject thresholds.")

    thresholds = subject_thresholds(clean, expected_fpr)

    print("Scoring per-type anomalous windows + evaluating...")
    results = evaluate(trainer, data_dir, clean, mixed, truth, thresholds, subjects=held)
    print_metrics(results, expected_fpr)

    report_dir = get_report_dir(args.model)
    write_yaml(report_dir / EVAL_REPORT, {
        'shows': f"Detector evaluation for {args.model} (report Sec. 5.4): precision/"
                 f"recall/F1/accuracy and clean false-positive rate against the true "
                 f"mixed-window labels, plus per-anomaly-kind recall, on held-out subjects.",
        'measured_on': {
            'calibration_subjects': train_ids,
            'eval_subjects': held_out,
            'note': "the expected FPR is calibrated on the training subjects; every "
                    "metric here is scored on the held-out subjects, so the numbers are "
                    "generalization to an unseen user.",
        },
        'selection': {'expected_fpr': expected_fpr},
        'headline': results['detector'],
        'per_kind': results['per_kind'],
        'n_windows': results['n_windows'],
        'gt_anomaly_rate': results['gt_anomaly_rate'],
        'pred_anomaly_rate': results['pred_anomaly_rate'],
        'source': {'reproducible': True},
    })
