import argparse
import json
from pathlib import Path
import numpy as np

from common.config import RESULTS_DIR, DATASETS_DIR
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer
from ml.data import (
    SUBJECTS_SUBDIR, ANOMALOUS_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR, ANOMALY_KINDS, 
    conditional_windows, get_sorted_paths
)
from ml.metrics import best_threshold, classification_report
from .common.post_train import get_report_dir, AE_TEST_REPORT
from .common.autoencoders import load_autoencoder, window_errors

def score_mixed(trainer: AutoencoderTrainer, data_dir: Path):
    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = get_sorted_paths(mixed_dir)
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    all_err, all_lbl = [], []
    for d in subject_dirs:
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=mixed_dir)
        truth = np.load(feature_dir / sid / 'labels.npy').reshape(-1)
        err = window_errors(trainer.model, signal, cond, window, len(truth))
        all_err.append(err)
        all_lbl.append(truth[:len(err)])
    return np.concatenate(all_err), np.concatenate(all_lbl)


def score_bvp_dir(trainer: AutoencoderTrainer, data_dir: Path,
                  bvp_dir: Path | None) -> np.ndarray:
    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR

    all_err = []
    for d in get_sorted_paths(subjects_dir):
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=bvp_dir)
        if len(cond) > 0:
            all_err.append(window_errors(trainer.model, signal, cond, window, len(cond)))
    return np.concatenate(all_err) if all_err else np.empty(0, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a trained autoencoder as an anomaly detector: pick the '
                    'F1-optimal reconstruction-error threshold on the realistic mixed '
                    'set, then report overall metrics, per-anomaly-kind recall (scored '
                    'on the per-type anomalous-signals/), and the clean-signal false '
                    'positive rate. Writes everything to results/<model>/reports/; '
                    'distill_labels.py reads the threshold from there.')
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to test')
    args = parser.parse_args()

    trainer = load_autoencoder(args.model)

    print("Scoring mixed-anomaly windows (threshold + overall metrics)...")
    errors, truth = score_mixed(trainer, DATASETS_DIR)
    thr, f1 = best_threshold(errors, truth, objective='f1')
    pred = errors > thr
    report = classification_report(pred, truth)

    print("Scoring clean windows (false-positive rate)...")
    clean_err = score_bvp_dir(trainer, DATASETS_DIR, None)
    clean_fpr = float((clean_err > thr).mean()) if len(clean_err) else 0.0

    print("Scoring per-type anomalous windows...")
    anomalous_dir = DATASETS_DIR / ANOMALOUS_SUBDIR
    kind_recall = {}
    for name in ANOMALY_KINDS:
        errs = score_bvp_dir(trainer, DATASETS_DIR, anomalous_dir / name)
        kind_recall[name] = {'recall': float((errs > thr).mean()) if len(errs) else None,
                             'count': int(len(errs))}

    print(f"\nthreshold={thr:.6f} (F1-optimal)")
    print(f"accuracy={report['accuracy']:.4f} precision={report['precision']:.4f} "
          f"recall={report['recall']:.4f} f1={report['f1']:.4f}")
    print(f"ground-truth anomaly rate={truth.mean():.1%}  predicted rate={pred.mean():.1%}")
    print(f"clean-signal false-positive rate={clean_fpr:.4f}")
    print("\nrecall by anomaly kind (scored on per-type anomalous-signals/):")
    for name, stats in kind_recall.items():
        r = 'n/a' if stats['recall'] is None else f"{stats['recall']:.4f}"
        print(f"  {name:<9} recall={r}  ({stats['count']} windows)")

    results = {
        'model': args.model,
        'threshold': thr,
        'objective': 'f1',
        'n_windows': int(len(truth)),
        'gt_anomaly_rate': float(truth.mean()),
        'pred_anomaly_rate': float(pred.mean()),
        'accuracy': report['accuracy'],
        'precision': report['precision'],
        'recall': report['recall'],
        'f1': report['f1'],
        'clean_false_positive_rate': clean_fpr,
        'per_kind_recall': kind_recall,
    }

    report_dir = get_report_dir(RESULTS_DIR / args.model)
    report_path = report_dir / AE_TEST_REPORT
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote evaluation report to {report_path}")
