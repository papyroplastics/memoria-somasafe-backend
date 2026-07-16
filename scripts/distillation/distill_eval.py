"""
Evaluate the autoencoder anomaly detector against the synthetic ground truth — the
scientific / testing step, with no restriction on what data it reads. Uses the same
expected FPR (distill_calibrate.py) and per-subject thresholds a client would
(distill_labels.py), then scores the detector against the true mixed-window labels and the
per-type anomalous-signals/ sets: precision/recall/F1, per-anomaly-kind recall, and the
empirical clean false-positive rate. The spectral baseline is measured the same way
alongside it, so the learned teacher can be read against a hand-crafted index. Writes the
metrics to results/<model>/.
"""


import argparse
import json
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.preprocessing import ANOMALOUS_SUBDIR, ANOMALY_KINDS
from ml.metrics import classification_report
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights
from ..common.reports import get_report_dir
from ..common.scoring import (
    SCORE_NAMES, DETECTOR, BASELINE, load_expected_fpr, subject_thresholds, pooled_flags,
    score_dir_by_subject, score_mixed_by_subject, load_mixed_truth,
)

EVAL_REPORT = 'distill_eval.json'   # detector metrics, from this script


def evaluate(trainer, data_dir: Path, clean: dict[str, dict[str, np.ndarray]],
             mixed: dict[str, dict[str, np.ndarray]], truth: dict[str, np.ndarray],
             thresholds: dict[str, dict[str, float]]) -> dict:
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])

    per_score = {}
    for n in SCORE_NAMES:
        rep = classification_report(pooled_flags(mixed, thresholds, n), pooled_truth)
        per_score[n] = {
            'precision': rep['precision'], 'recall': rep['recall'], 'f1': rep['f1'],
            'accuracy': rep['accuracy'],
            'clean_fpr': float(pooled_flags(clean, thresholds, n).mean()),
        }

    anomalous_dir = data_dir / ANOMALOUS_SUBDIR
    per_kind = {}
    for name in ANOMALY_KINDS:
        sc = score_dir_by_subject(trainer, data_dir, anomalous_dir / name)
        c = sum(len(v[DETECTOR]) for v in sc.values())
        per_kind[name] = {
            'count': c,
            'by_score': {n: (float(pooled_flags(sc, thresholds, n).mean()) if c else None)
                         for n in SCORE_NAMES},
        }

    return {
        'n_windows': int(len(pooled_truth)),
        'gt_anomaly_rate': float(pooled_truth.mean()),
        'pred_anomaly_rate': float(pooled_flags(mixed, thresholds).mean()),
        'detector': per_score[DETECTOR],
        'baseline': per_score[BASELINE],
        'per_kind': per_kind,
    }


def print_metrics(results: dict, expected_fpr: float):
    d = results['detector']
    print(f"\nexpected_fpr={expected_fpr:.4f}")
    print(f"detector ({DETECTOR}): accuracy={d['accuracy']:.4f} precision={d['precision']:.4f} "
          f"recall={d['recall']:.4f} f1={d['f1']:.4f} clean_fpr={d['clean_fpr']:.4f}")
    b = results['baseline']
    print(f"baseline ({BASELINE}): accuracy={b['accuracy']:.4f} precision={b['precision']:.4f} "
          f"recall={b['recall']:.4f} f1={b['f1']:.4f} clean_fpr={b['clean_fpr']:.4f}")
    print(f"ground-truth anomaly rate={results['gt_anomaly_rate']:.1%}  "
          f"predicted rate={results['pred_anomaly_rate']:.1%}")

    print("\nrecall by anomaly kind (scored on per-type anomalous-signals/):")
    print(f"  {'kind':<9} " + "  ".join(f"{n:>9}" for n in SCORE_NAMES))
    for name, stats in results['per_kind'].items():
        parts = "  ".join('      n/a' if stats['by_score'][n] is None
                          else f"{stats['by_score'][n]:>9.4f}" for n in SCORE_NAMES)
        print(f"  {name:<9} {parts}  ({stats['count']} windows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to evaluate')
    args = parser.parse_args()

    data_dir = DATASETS_DIR

    expected_fpr = load_expected_fpr(args.model)

    trainer = MODELS[args.model].build_trainer(data_dir)
    trainer.model.restore(load_trainable_weights(MODELS_DIR / args.model / 'trainable.tflite'))
    assert isinstance(trainer, AutoencoderTrainer)

    print("Scoring mixed-anomaly windows...")
    mixed = score_mixed_by_subject(trainer, data_dir)
    truth = load_mixed_truth(data_dir, mixed)
    print("Scoring clean windows (sets each subject's thresholds)...")
    clean = score_dir_by_subject(trainer, data_dir, None)
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; "
                         "cannot derive per-subject thresholds.")

    thresholds = subject_thresholds(clean, expected_fpr)

    print("Scoring per-type anomalous windows + evaluating...")
    results = evaluate(trainer, data_dir, clean, mixed, truth, thresholds)
    print_metrics(results, expected_fpr)

    report_dir = get_report_dir(args.model)
    eval_path = report_dir / EVAL_REPORT
    eval_path.write_text(json.dumps(
        {'model': args.model, 'expected_fpr': expected_fpr, **results}, indent=2))
    print(f"\nWrote detector metrics to {eval_path}")
