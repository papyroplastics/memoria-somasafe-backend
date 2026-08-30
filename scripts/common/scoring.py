"""Per-window anomaly scoring: the autoencoder's reconstruction MSE, thresholded at the
``1 - expected_fpr`` quantile of clean-window scores, per subject or pooled. See
../../../shared/docs/anomalies-and-distillation.md for why the expected FPR is calibrated
on Youden's J."""

import numpy as np

import tensorflow as tf
from ml.metrics import classification_report
from ml.models.common import DescriptorAutoencoder, TrainableAutoencoder
from ml.preprocessing import BVP_WINDOW
from ml.sources.common import DataSource


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


def window_errors(model, windows: np.ndarray) -> np.ndarray:
    if not len(windows):
        return np.empty(0, dtype=np.float32)
    out = eval_padded(model, windows.astype(np.float32))
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


def scored_subjects(source: DataSource, subjects: set[str] | None = None) -> list[str]:
    return [sid for sid in source.subject_ids()
            if subjects is None or sid in subjects]


def datapoints(model: TrainableAutoencoder, source: DataSource, sid: str,
               variant: str) -> np.ndarray:
    """One subject's datapoints for a variant, in whatever the model eats."""
    if isinstance(model, DescriptorAutoencoder):
        return source.descriptors(sid, variant, BVP_WINDOW, BVP_WINDOW)
    return source.signal_windows(sid, variant, model.seq_len, model.seq_len)


def score_subjects(model: TrainableAutoencoder, source: DataSource, variant: str,
                   subjects: set[str] | None = None) -> dict[str, np.ndarray]:
    """Returns per-window reconstruction error for one signal variant."""
    return {sid: window_errors(model, datapoints(model, source, sid, variant))
            for sid in scored_subjects(source, subjects)}


def mixed_truth(source: DataSource,
                subjects: set[str] | None = None) -> dict[str, np.ndarray]:
    return {sid: source.window_labels(sid)
            for sid in scored_subjects(source, subjects)}
