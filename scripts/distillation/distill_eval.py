"""
Evaluate the autoencoder anomaly detector against the synthetic ground truth — the
scientific / testing step, with no restriction on what data it reads. Uses the same
budgets (distill_calibrate.py) and per-subject thresholds a client would (distill_labels.py),
then scores the OR detector against the true mixed-window labels and the per-type
anomalous-signals/ sets: OR-combined + per-score precision/recall/F1, per-anomaly-kind
recall, and the clean false-positive rate. Writes the metrics to results/<model>/.
"""


import argparse
import json
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR
from ml.data import ANOMALOUS_SUBDIR, ANOMALY_KINDS, BVP_WINDOW, WINDOW_SECONDS
from ml.metrics import classification_report
from ml.model_list import MODELS
from ..common.autoencoders import load_autoencoder
from ..common.scoring import (
    SCORE_NAMES, subject_thresholds, pooled_predict,
    score_dir_by_subject, score_mixed_by_subject, load_mixed_truth,
)
from ..common.post_train import get_report_dir, load_budgets, EVAL_REPORT


def evaluate(trainer, data_dir: Path, clean: dict[str, dict[str, np.ndarray]],
             mixed: dict[str, dict[str, np.ndarray]], truth: dict[str, np.ndarray],
             thresholds: dict[str, dict[str, float]]) -> dict:
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])
    pred = pooled_predict(mixed, thresholds)
    combined = classification_report(pred, pooled_truth)

    def pooled_single(by_subject: dict[str, dict[str, np.ndarray]], name: str) -> np.ndarray:
        return np.concatenate([by_subject[sid][name] > thresholds[sid][name]
                               for sid in by_subject])

    per_score = {}
    for n in SCORE_NAMES:
        rep = classification_report(pooled_single(mixed, n), pooled_truth)
        per_score[n] = {
            'precision': rep['precision'], 'recall': rep['recall'], 'f1': rep['f1'],
            'clean_fpr': float(pooled_single(clean, n).mean()),
        }

    anomalous_dir = data_dir / ANOMALOUS_SUBDIR
    per_kind = {}
    for name in ANOMALY_KINDS:
        sc = score_dir_by_subject(trainer, data_dir, anomalous_dir / name)
        c = sum(len(v['recon']) for v in sc.values())
        per_kind[name] = {
            'count': c,
            'combined_recall': float(pooled_predict(sc, thresholds).mean()) if c else None,
            'by_score': {n: (float(pooled_single(sc, n).mean()) if c else None)
                         for n in SCORE_NAMES},
        }

    return {
        'n_windows': int(len(pooled_truth)),
        'gt_anomaly_rate': float(pooled_truth.mean()),
        'pred_anomaly_rate': float(pred.mean()),
        'combined': {
            'precision': combined['precision'], 'recall': combined['recall'],
            'f1': combined['f1'], 'accuracy': combined['accuracy'],
            'clean_false_positive_rate': float(pooled_predict(clean, thresholds).mean()),
        },
        'per_score': per_score,
        'per_kind': per_kind,
    }


def print_metrics(results: dict, budgets: dict[str, float]):
    c = results['combined']
    print(f"\nbudgets:  " + "  ".join(f"{n}={budgets[n]:.4f}" for n in SCORE_NAMES))
    print(f"combined: accuracy={c['accuracy']:.4f} precision={c['precision']:.4f} "
          f"recall={c['recall']:.4f} f1={c['f1']:.4f}")
    print(f"ground-truth anomaly rate={results['gt_anomaly_rate']:.1%}  "
          f"predicted rate={results['pred_anomaly_rate']:.1%}")
    print(f"clean-signal false-positive rate={c['clean_false_positive_rate']:.4f}")

    print("\nper score (on mixed set / clean):")
    for n in SCORE_NAMES:
        s = results['per_score'][n]
        print(f"  {n:<9} recall={s['recall']:.4f} precision={s['precision']:.4f} "
              f"f1={s['f1']:.4f}  clean_fpr={s['clean_fpr']:.4f}")

    print("\nrecall by anomaly kind (scored on per-type anomalous-signals/):")
    print(f"  {'kind':<9} {'combined':>9}   " + "  ".join(SCORE_NAMES))
    for name, stats in results['per_kind'].items():
        cr = 'n/a' if stats['combined_recall'] is None else f"{stats['combined_recall']:.4f}"
        parts = "  ".join('n/a' if stats['by_score'][n] is None else f"{stats['by_score'][n]:.4f}"
                          for n in SCORE_NAMES)
        print(f"  {name:<9} {cr:>9}   {parts}  ({stats['count']} windows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to evaluate')
    args = parser.parse_args()

    data_dir = DATASETS_DIR

    budgets = load_budgets(args.model)
    # batch_size=1 to match exactly the thresholds distill_labels/a client would use.
    trainer = load_autoencoder(args.model, batch_size=1)

    window = trainer.window_size
    if window != BVP_WINDOW:
        raise SystemExit(
            f"model window ({window} samples) does not match the {WINDOW_SECONDS}s feature "
            f"window ({BVP_WINDOW} samples) used to build mixed-features. Align the window.")

    print("Scoring mixed-anomaly windows...")
    mixed = score_mixed_by_subject(trainer, data_dir)
    truth = load_mixed_truth(data_dir, mixed)
    print("Scoring clean windows (sets each subject's thresholds)...")
    clean = score_dir_by_subject(trainer, data_dir, None)
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; "
                         "cannot derive per-subject thresholds.")

    thresholds = subject_thresholds(clean, budgets)

    print("Scoring per-type anomalous windows + evaluating...")
    results = evaluate(trainer, data_dir, clean, mixed, truth, thresholds)
    print_metrics(results, budgets)

    report_dir = get_report_dir(args.model)
    eval_path = report_dir / EVAL_REPORT
    eval_path.write_text(json.dumps({'model': args.model, 'budgets': budgets, **results}, indent=2))
    print(f"\nWrote detector metrics to {eval_path}")
