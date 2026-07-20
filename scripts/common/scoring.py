"""Per-window anomaly scoring: the autoencoder's reconstruction MSE, thresholded at the
``1 - expected_fpr`` quantile of clean-window scores — per subject, or over all subjects
pooled. See ../../../shared/docs/anomalies-and-distillation.md for why the expected FPR is
calibrated on Youden's J.
"""

from pathlib import Path

import numpy as np

import tensorflow as tf
from ml.preprocessing import MIXED_FEATURE_SUBDIR, get_sorted_paths
from ml.loading import load_signal, window_count
from ml.metrics import classification_report
from ml.models.common import AutoencoderTrainer


def eval_padded(model, *arrays: np.ndarray) -> dict[str, np.ndarray]:
    n = len(arrays[0])
    if n == 0:
        return {}

    batch_size = model.batch_size
    pad = (-n) % batch_size
    if pad:
        arrays = tuple(np.concatenate([a, np.repeat(a[-1:], pad, axis=0)]) for a in arrays)

    chunks = []
    for start in range(0, n + pad, batch_size):
        batch = [tf.constant(a[start:start + batch_size], dtype=tf.float32) for a in arrays]
        chunks.append({k: np.asarray(v) for k, v in model.eval(*batch).items()})
    return {k: np.concatenate([c[k] for c in chunks])[:n] for k in chunks[0]}


def window_errors(model, signal: np.ndarray, window: int, n_windows: int) -> np.ndarray:
    n_windows = min(n_windows, len(signal) // window)
    if n_windows <= 0:
        return np.empty(0, dtype=np.float32)
    windows = (signal[:n_windows * window]
               .reshape(n_windows, window, model.n_signals).astype(np.float32))
    out = eval_padded(model, windows)
    return out['error'].reshape(-1).astype(np.float32)


def clean_threshold(clean_score: np.ndarray, expected_fpr: float) -> float:
    return float(np.quantile(clean_score, 1.0 - expected_fpr))


def subject_thresholds(clean: dict[str, np.ndarray], expected_fpr: float) -> dict[str, float]:
    return {sid: clean_threshold(sc, expected_fpr) for sid, sc in clean.items()}


def global_thresholds(clean: dict[str, np.ndarray], expected_fpr: float) -> dict[str, float]:
    thr = clean_threshold(np.concatenate(list(clean.values())), expected_fpr)
    return {sid: thr for sid in clean}


def pooled_flags(scores: dict[str, np.ndarray],
                 thresholds: dict[str, float]) -> np.ndarray:
    return np.concatenate([scores[sid] > thresholds[sid] for sid in scores])


def _fpr_row(clean: dict[str, np.ndarray], mixed: dict[str, np.ndarray],
             truth: dict[str, np.ndarray], f: float, thresholds_fn) -> dict:
    thr = thresholds_fn(clean, f)
    fpr = float(pooled_flags(clean, thr).mean())

    pooled_truth = np.concatenate([truth[sid] for sid in mixed])
    rep = classification_report(pooled_flags(mixed, thr), pooled_truth)

    rep['expected_fpr'] = f
    rep['clean_fpr'] = fpr
    rep['youden_j'] = rep['recall'] - fpr

    return rep


def sweep_expected_fpr(clean: dict[str, np.ndarray], mixed: dict[str, np.ndarray],
                       truth: dict[str, np.ndarray], grid,
                       thresholds_fn=subject_thresholds) -> list[dict]:
    return [_fpr_row(clean, mixed, truth, f, thresholds_fn) for f in sorted(grid)]


def calibrate_expected_fpr(clean: dict[str, np.ndarray], mixed: dict[str, np.ndarray],
                           truth: dict[str, np.ndarray], step: float = 0.0025,
                           thresholds_fn=subject_thresholds) -> float:
    grid = np.round(np.arange(step, 1.0 + step / 2, step), 6).tolist()
    rows = sweep_expected_fpr(clean, mixed, truth, grid, thresholds_fn)
    best = max(range(len(rows)), key=lambda i: rows[i]['youden_j'])
    return float(rows[best]['expected_fpr'])


def score_dir_by_subject(trainer: AutoencoderTrainer, signal_dir: Path,
                         subjects: set[str] | None = None) -> dict[str, np.ndarray]:
    window = trainer.model.seq_len
    subject_dirs = get_sorted_paths(signal_dir)
    if not subject_dirs:
        raise SystemExit(f"{signal_dir} is empty. Run get_dataset.py first.")

    out: dict[str, np.ndarray] = {}
    for d in subject_dirs:
        sid = d.name
        if subjects is not None and sid not in subjects:
            continue
        signal = load_signal(signal_dir, sid)
        count = window_count(signal, window)
        out[sid] = window_errors(trainer.model, signal, window, count)
    return out


def load_mixed_truth(data_dir: Path) -> dict[str, np.ndarray]:
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR
    return {d.name: np.load(d / 'labels.npy').reshape(-1) for d in feature_dir.glob('S*')}
