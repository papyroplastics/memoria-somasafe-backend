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
from ml.data import (SUBJECTS_SUBDIR, ANOMALOUS_SUBDIR, FEATURE_SUBDIR,
                     ANOMALY_KINDS, stacked_signal, norm_stats, normalize)
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


def score_subjects(trainer: AutoencoderTrainer, data_dir: Path):
    """Score every subject's synthetic-anomaly windows by reconstruction error.
    Returns (errors, truth, kinds) concatenated across subjects, all per-window and
    aligned 1:1 with the feature/distillation grid."""
    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR
    anomalous_dir = data_dir / ANOMALOUS_SUBDIR
    feature_dir = data_dir / FEATURE_SUBDIR

    subject_dirs = sorted(anomalous_dir.glob('S*'))
    if not subject_dirs:
        raise SystemExit(f"{anomalous_dir} is empty. Run get_dataset.py first.")

    mean, std = norm_stats(subjects_dir)

    all_err, all_lbl, all_kind = [], [], []
    print("Scoring windows by reconstruction error:")
    for d in subject_dirs:
        sid = d.name
        signal = normalize(stacked_signal(subjects_dir, sid, anomalous_dir=anomalous_dir),
                           mean, std)
        truth = np.load(feature_dir / sid / 'labels.npy').reshape(-1)
        kinds = np.load(anomalous_dir / sid / 'kinds.npy').reshape(-1)[:len(truth)]
        errs = window_errors(trainer.model, signal, window, len(truth))
        all_err.append(errs)
        all_lbl.append(truth)
        all_kind.append(kinds)
        print(f"  {sid}: {len(truth)} windows")

    return (np.concatenate(all_err), np.concatenate(all_lbl), np.concatenate(all_kind))


def per_kind_recall(pred: np.ndarray, kinds: np.ndarray) -> dict[str, dict[str, float]]:
    """Recall (detection rate) for each anomaly kind separately — shows which kinds
    reconstruction error is blind to."""
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(ANOMALY_KINDS, start=1):
        mask = kinds == idx
        count = int(mask.sum())
        recall = float(pred[mask].mean()) if count else None
        out[name] = {'recall': recall, 'count': count}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate a trained autoencoder as an anomaly detector: score the '
                    'synthetic-anomaly windows by reconstruction error, pick the '
                    'F1-optimal threshold against the ground-truth labels, and write '
                    'the threshold + metrics (overall and per anomaly kind) to '
                    'results/<model>/reports/. distill_labels.py reads the threshold '
                    'from there.')
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to test')
    args = parser.parse_args()

    trainer = load_autoencoder(args.model, DATASETS_DIR)
    errors, truth, kinds = score_subjects(trainer, DATASETS_DIR)

    thr, f1 = best_threshold(errors, truth, objective='f1')
    pred = errors > thr
    report = classification_report(pred, truth)
    kind_recall = per_kind_recall(pred, kinds)

    print(f"\nthreshold={thr:.6f} (F1-optimal)")
    print(f"accuracy={report['accuracy']:.4f} precision={report['precision']:.4f} "
          f"recall={report['recall']:.4f} f1={report['f1']:.4f}")
    print(f"ground-truth anomaly rate={truth.mean():.1%}  predicted rate={pred.mean():.1%}")
    print("\nrecall by anomaly kind:")
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
        'per_kind_recall': kind_recall,
    }

    report_dir = get_report_dir(RESULTS_DIR / args.model)
    report_path = report_dir / AE_TEST_REPORT
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote evaluation report to {report_path}")
