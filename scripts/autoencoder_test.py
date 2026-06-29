"""
Evaluate a trained autoencoder as an anomaly detector. Thresholds 
each score (reconstruction error + spectral / beat-interval rhythm 
indices) at the --target-fpr quantile of its clean-window distribution, 
then reports the OR-combined metrics, each score on its own, per-anomaly-
kind recall (scored on the per-type anomalous-signals/, broken down 
by score) and the clean-signal false-positive rate. Writes the 
thresholds + metrics to results/<model>/reports/; distill_labels.py 
reads the thresholds from there.
"""


import argparse
import json
from pathlib import Path
import numpy as np

from common.config import RESULTS_DIR, DATASETS_DIR
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer
from ml.data import (
    CLEAN_SUBDIR, ANOMALOUS_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR, ANOMALY_KINDS,
    conditional_windows, get_sorted_paths
)
from ml.metrics import classification_report
from .common.post_train import get_report_dir, AE_TEST_REPORT
from .common.autoencoders import load_autoencoder
from .common.scoring import SCORE_NAMES, score_windows, pick_thresholds, predict


def _concat(per_subject: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {n: (np.concatenate([s[n] for s in per_subject]) if per_subject
                else np.empty(0, dtype=np.float32)) for n in SCORE_NAMES}


def score_mixed(trainer: AutoencoderTrainer, data_dir: Path):
    """Score the realistic mixed-anomaly windows; returns (scores, truth) aligned 1:1
    with the mixed-feature labels."""
    window = trainer.window_size
    subjects_dir = data_dir / CLEAN_SUBDIR
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = get_sorted_paths(mixed_dir)
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    per_subject, all_lbl = [], []
    for d in subject_dirs:
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=mixed_dir)
        truth = np.load(feature_dir / sid / 'labels.npy').reshape(-1)
        scores = score_windows(trainer.model, signal, cond, window, len(truth))
        per_subject.append(scores)
        all_lbl.append(truth[:len(scores['recon'])])
    return _concat(per_subject), np.concatenate(all_lbl)


def score_dir(trainer: AutoencoderTrainer, data_dir: Path,
              bvp_dir: Path | None) -> dict[str, np.ndarray]:
    """Score every non-overlapping window across all subjects, taking BVP from
    ``bvp_dir`` (None = clean clean-signals) and ACC from clean-signals — a single
    anomaly kind (every window anomalous) or the clean baseline."""
    window = trainer.window_size
    subjects_dir = data_dir / CLEAN_SUBDIR

    per_subject = []
    for d in get_sorted_paths(subjects_dir):
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=bvp_dir)
        if len(cond) > 0:
            per_subject.append(score_windows(trainer.model, signal, cond, window, len(cond)))
    return _concat(per_subject)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to test')
    parser.add_argument('--target-fpr', type=float, default=0.02,
                        help='Per-score clean-window false-positive rate the thresholds '
                             'target (default: 0.02). Combined FPR is ~3x this at most.')
    args = parser.parse_args()

    trainer = load_autoencoder(args.model)

    print("Scoring mixed-anomaly windows...")
    scores, truth = score_mixed(trainer, DATASETS_DIR)

    print("Scoring clean windows (sets the thresholds)...")
    clean = score_dir(trainer, DATASETS_DIR, None)
    n_clean = len(clean['recon'])
    if not n_clean:
        raise SystemExit("no clean windows to set thresholds from.")

    thresholds = pick_thresholds(clean, args.target_fpr)
    pred = predict(scores, thresholds)
    combined = classification_report(pred, truth)
    clean_fpr = float(predict(clean, thresholds).mean())

    # Each score on its own: its mixed-set precision/recall/f1 and its clean FPR.
    per_score = {}
    for n in SCORE_NAMES:
        rep = classification_report(scores[n] > thresholds[n], truth)
        per_score[n] = {
            'threshold': thresholds[n],
            'precision': rep['precision'], 'recall': rep['recall'], 'f1': rep['f1'],
            'clean_fpr': float((clean[n] > thresholds[n]).mean()) if n_clean else 0.0,
        }

    print("Scoring per-type anomalous windows...")
    anomalous_dir = DATASETS_DIR / ANOMALOUS_SUBDIR
    per_kind = {}
    for name in ANOMALY_KINDS:
        sc = score_dir(trainer, DATASETS_DIR, anomalous_dir / name)
        c = len(sc['recon'])
        per_kind[name] = {
            'count': c,
            'combined_recall': float(predict(sc, thresholds).mean()) if c else None,
            'by_score': {n: (float((sc[n] > thresholds[n]).mean()) if c else None)
                         for n in SCORE_NAMES},
        }

    print(f"\nthresholds (per score, at {args.target_fpr:.1%} clean FPR each):  "
          + "  ".join(f"{n}={thresholds[n]:.6f}" for n in SCORE_NAMES))
    print(f"combined: accuracy={combined['accuracy']:.4f} precision={combined['precision']:.4f} "
          f"recall={combined['recall']:.4f} f1={combined['f1']:.4f}")
    print(f"ground-truth anomaly rate={truth.mean():.1%}  predicted rate={pred.mean():.1%}")
    print(f"clean-signal false-positive rate={clean_fpr:.4f}")

    print("\nper score (on mixed set / clean):")
    for n in SCORE_NAMES:
        s = per_score[n]
        print(f"  {n:<9} recall={s['recall']:.4f} precision={s['precision']:.4f} "
              f"f1={s['f1']:.4f}  clean_fpr={s['clean_fpr']:.4f}")

    print("\nrecall by anomaly kind (scored on per-type anomalous-signals/):")
    header = "  ".join(f"{n}" for n in SCORE_NAMES)
    print(f"  {'kind':<9} {'combined':>9}   {header}")
    for name, stats in per_kind.items():
        cr = 'n/a' if stats['combined_recall'] is None else f"{stats['combined_recall']:.4f}"
        parts = "  ".join('n/a' if stats['by_score'][n] is None else f"{stats['by_score'][n]:.4f}"
                          for n in SCORE_NAMES)
        print(f"  {name:<9} {cr:>9}   {parts}  ({stats['count']} windows)")

    results = {
        'model': args.model,
        'objective': 'clean_fpr_quantile',
        'target_fpr': args.target_fpr,
        'thresholds': thresholds,
        'n_windows': int(len(truth)),
        'gt_anomaly_rate': float(truth.mean()),
        'pred_anomaly_rate': float(pred.mean()),
        'combined': {
            'precision': combined['precision'], 'recall': combined['recall'],
            'f1': combined['f1'], 'accuracy': combined['accuracy'],
            'clean_false_positive_rate': clean_fpr,
        },
        'per_score': per_score,
        'per_kind': per_kind,
    }

    report_dir = get_report_dir(RESULTS_DIR / args.model)
    report_path = report_dir / AE_TEST_REPORT
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote evaluation report to {report_path}")
