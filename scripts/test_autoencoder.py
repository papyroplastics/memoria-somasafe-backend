import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from common.config import RESULTS_DIR, DATASETS_DIR, SEED
from ml.saving import load_trainable_weights
from ml.model_list import MODELS, build_trainer
from ml.models.common import AutoencoderTrainer
from ml.models.cond_lstm_autoencoder import ConditionalAutoencoderTrainer
from ml.data import (SUBJECTS_SUBDIR, ANOMALOUS_SUBDIR, MIXED_SUBDIR,
                     MIXED_FEATURE_SUBDIR, ANOMALY_KINDS,
                     stacked_signal, norm_stats, normalize)
from ml.metrics import best_threshold, classification_report
from .common.post_train import get_report_dir, AE_TEST_REPORT


def window_errors(model, signal: np.ndarray, window: int, n_windows: int) -> np.ndarray:
    """Reconstruction error for the first ``n_windows`` non-overlapping windows of a
    normalized ``[BVP, ACC]`` signal — the autoencoder's per-window anomaly score.
    Built with a batch-size-1 model so each window scores independently."""
    errors = np.empty(n_windows, dtype=np.float32)
    for w in range(n_windows):
        s = w * window
        win = signal[s:s + window]
        out = model.eval(win[None].astype(np.float32))
        errors[w] = float(out['error'][0])
    return errors


def load_autoencoder(model_name: str, data_dir: Path) -> AutoencoderTrainer:
    """Build a batch-size-1 trainer for a non-conditional autoencoder and restore
    its trained weights from results/<model>/trainable.tflite."""
    trainer = build_trainer(model_name, data_dir, SEED, batch_size=1)
    if not isinstance(trainer, AutoencoderTrainer) or isinstance(trainer, ConditionalAutoencoderTrainer):
        raise SystemExit(
            f"'{model_name}' is not a non-conditional autoencoder; testing needs "
            f"one (lstm-ae, gru-ae, cnn-ae).")

    weights_path = RESULTS_DIR / model_name / 'trainable.tflite'
    if not weights_path.exists():
        raise SystemExit(f"trained model not found at {weights_path}. Train '{model_name}' first.")
    trainer.model.restore(tf.constant(load_trainable_weights(weights_path)))
    return trainer


def score_mixed(trainer: AutoencoderTrainer, data_dir: Path, mean, std):
    """Score the realistic mixed-anomaly windows; returns (errors, truth) per window,
    aligned 1:1 with the mixed-feature labels."""
    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = sorted(mixed_dir.glob('S*'))
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    all_err, all_lbl = [], []
    for d in subject_dirs:
        sid = d.name
        signal = normalize(stacked_signal(subjects_dir, sid, anomalous_dir=mixed_dir), mean, std)
        truth = np.load(feature_dir / sid / 'labels.npy').reshape(-1)
        all_err.append(window_errors(trainer.model, signal, window, len(truth)))
        all_lbl.append(truth)
    return np.concatenate(all_err), np.concatenate(all_lbl)


def score_bvp_dir(trainer: AutoencoderTrainer, data_dir: Path,
                  bvp_dir: Path | None, mean, std) -> np.ndarray:
    """Score every non-overlapping window across all subjects, taking BVP from
    ``bvp_dir`` (None = clean subject-signals) and ACC from subject-signals. Used for
    a single anomaly kind (every window anomalous) or the clean baseline."""
    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR

    all_err = []
    for d in sorted(subjects_dir.glob('S*')):
        sid = d.name
        signal = normalize(stacked_signal(subjects_dir, sid, anomalous_dir=bvp_dir), mean, std)
        n = (len(signal) - window) // window + 1
        if n > 0:
            all_err.append(window_errors(trainer.model, signal, window, n))
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

    trainer = load_autoencoder(args.model, DATASETS_DIR)
    mean, std = norm_stats(DATASETS_DIR / SUBJECTS_SUBDIR)

    print("Scoring mixed-anomaly windows (threshold + overall metrics)...")
    errors, truth = score_mixed(trainer, DATASETS_DIR, mean, std)
    thr, f1 = best_threshold(errors, truth, objective='f1')
    pred = errors > thr
    report = classification_report(pred, truth)

    print("Scoring clean windows (false-positive rate)...")
    clean_err = score_bvp_dir(trainer, DATASETS_DIR, None, mean, std)
    clean_fpr = float((clean_err > thr).mean()) if len(clean_err) else 0.0

    print("Scoring per-type anomalous windows...")
    anomalous_dir = DATASETS_DIR / ANOMALOUS_SUBDIR
    kind_recall = {}
    for name in ANOMALY_KINDS:
        errs = score_bvp_dir(trainer, DATASETS_DIR, anomalous_dir / name, mean, std)
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
