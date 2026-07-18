"""Per-window anomaly scoring shared by the distillation scripts.

The detector is the autoencoder's reconstruction MSE (``recon``), thresholded per
subject: the threshold is the ``1 - expected_fpr`` quantile of that subject's own
clean-window scores, so a subject-specific error scale gives a uniform per-subject
false-alarm rate instead of one dominated by the noisiest subjects. The server calibrates
the ``expected_fpr`` — a single global number, the only thing calibration picks — and each
client derives its own threshold from it. Because each threshold is a quantile of the
subject's own clean scores, that fraction of clean windows lies above it by definition: the
parameter *is* the false-alarm rate, not a proxy for it. The rate measured on a given set
is its *empirical* FPR (``clean_fpr``).

See ../../../shared/docs/anomalies-and-distillation.md for why the expected FPR is
calibrated on Youden's J.
"""

from pathlib import Path

import numpy as np

import tensorflow as tf
from ml.preprocessing import CLEAN_SUBDIR, MIXED_FEATURE_SUBDIR, get_sorted_paths
from ml.loading import load_signal, window_count, holdout
from ml.metrics import classification_report
from ml.models.common import AutoencoderTrainer

DETECTOR = 'recon'       # the autoencoder score the detector and the labels are built on


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


def score_windows(model, signal: np.ndarray, window: int,
                  n_windows: int) -> dict[str, np.ndarray]:
    return {DETECTOR: window_errors(model, signal, window, n_windows)}


def clean_threshold(clean_score: np.ndarray, expected_fpr: float) -> float:
    return float(np.quantile(clean_score, 1.0 - expected_fpr))


def subject_thresholds(clean: dict[str, dict[str, np.ndarray]],
                       expected_fpr: float) -> dict[str, dict[str, float]]:
    """Each subject's detector threshold at the shared global expected FPR."""
    return {sid: {DETECTOR: clean_threshold(sc[DETECTOR], expected_fpr)}
            for sid, sc in clean.items()}


def _fpr_row(clean: dict[str, dict[str, np.ndarray]],
            mixed: dict[str, dict[str, np.ndarray]],
            truth: dict[str, np.ndarray], f: float) -> dict:
    thr = {sid: clean_threshold(clean[sid][DETECTOR], f) for sid in clean}
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])
    rep = classification_report(
        np.concatenate([mixed[sid][DETECTOR] > thr[sid] for sid in mixed]), pooled_truth)
    fpr = float(np.concatenate([clean[sid][DETECTOR] > thr[sid] for sid in clean]).mean())
    return {'expected_fpr': f, 'recall': rep['recall'], 'precision': rep['precision'],
            'f1': rep['f1'], 'clean_fpr': fpr, 'youden_j': rep['recall'] - fpr}


def sweep_expected_fpr(clean: dict[str, dict[str, np.ndarray]],
                       mixed: dict[str, dict[str, np.ndarray]],
                       truth: dict[str, np.ndarray], grid) -> list[dict]:
    return [_fpr_row(clean, mixed, truth, f) for f in sorted(grid)]


def calibrate_expected_fpr(clean: dict[str, dict[str, np.ndarray]],
                           mixed: dict[str, dict[str, np.ndarray]],
                           truth: dict[str, np.ndarray],
                           step: float = 0.0025) -> float:
    """Expected FPR maximizing Youden's J, by a dense grid argmax.

    J over the expected FPR is not unimodal — it rises to a peak and then rides a
    noisy, near-flat plateau — so a coarse scan plus ternary search is unsound (it
    overshoots the peak and drifts toward the middle of its bracket). Evaluating J on
    a fine grid and taking the argmax is both cheap (each row is numpy over
    already-computed scores) and robust. The grid is ascending, so ties resolve to the
    lowest FPR — the same J with fewer false alarms."""
    grid = np.round(np.arange(step, 1.0 + step / 2, step), 6).tolist()
    rows = sweep_expected_fpr(clean, mixed, truth, grid)
    best = max(range(len(rows)), key=lambda i: rows[i]['youden_j'])
    return float(rows[best]['expected_fpr'])


def pooled_flags(scores: dict[str, dict[str, np.ndarray]],
                 thresholds: dict[str, dict[str, float]],
                 name: str = DETECTOR) -> np.ndarray:
    return np.concatenate([scores[sid][name] > thresholds[sid][name] for sid in scores])


def split_subject_ids(data_dir: Path, n_eval: int) -> tuple[list[str], list[str]]:
    sids = [d.name for d in get_sorted_paths(data_dir / CLEAN_SUBDIR)]
    return holdout(sids, n_eval)


def score_dir_by_subject(trainer: AutoencoderTrainer, signal_dir: Path,
                         subjects: set[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    window = trainer.model.seq_len
    subject_dirs = get_sorted_paths(signal_dir)
    if not subject_dirs:
        raise SystemExit(f"{signal_dir} is empty. Run get_dataset.py first.")

    out: dict[str, dict[str, np.ndarray]] = {}
    for d in subject_dirs:
        sid = d.name
        if subjects is not None and sid not in subjects:
            continue
        signal = load_signal(signal_dir, sid)
        count = window_count(signal, window)
        if count > 0:
            out[sid] = score_windows(trainer.model, signal, window, count)
    return out


def load_mixed_truth(data_dir: Path, scores: dict[str, dict[str, np.ndarray]]
                     ) -> dict[str, np.ndarray]:
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR
    return {sid: np.load(feature_dir / sid / 'labels.npy').reshape(-1)[:len(sc[DETECTOR])]
            for sid, sc in scores.items()}
