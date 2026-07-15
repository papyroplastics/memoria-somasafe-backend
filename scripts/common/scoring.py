"""Per-window anomaly scoring shared by distill_calibrate.py and distill_labels.py.

The detector is an OR of independently-thresholded scores, each oriented
*higher = more anomalous*:

  - ``recon``    reconstruction MSE from the autoencoder — amplitude/integrity
                 anomalies (blowup, noise).
  - ``spectral`` in-band spectral entropy — irregular/smeared rhythm (afib), noise.
  - ``rr``       beat-interval coefficient of variation — irregular rhythm (afib).

Each score gets a per-subject threshold — the ``1 - budget`` quantile of that subject's
own clean-window scores — and a window is anomalous if *any* score crosses its threshold.
distill_calibrate.py picks the global per-score budgets; distill_labels.py turns them
into per-subject thresholds and soft labels.
"""

import json
from pathlib import Path

import numpy as np

import tensorflow as tf
from ml.preprocessing import (
    CLEAN_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR,
    conditional_windows, get_sorted_paths,
)
from ml.models.common import AutoencoderTrainer

from . import dsp
from .reports import get_report_dir

SCORE_NAMES = ('recon', 'spectral', 'rr')
CALIBRATION_REPORT = 'distill_calibration.json'   # budgets, from distill_calibrate


def eval_padded(model, *arrays: np.ndarray) -> dict[str, np.ndarray]:
    """Run ``model.eval`` over arrays whose length need not be a multiple of the model's
    batch size; returns each output tensor stacked over the rows, in order."""
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


def load_budgets(model_name: str) -> dict[str, float]:
    """The global per-score budgets picked by distill_calibrate.py."""
    report_path = get_report_dir(model_name) / CALIBRATION_REPORT
    if not report_path.exists():
        raise SystemExit(
            f"no calibration report at {report_path}. Run distill_calibrate '{model_name}' "
            f"first to pick the budgets.")
    return {k: float(v) for k, v in json.loads(report_path.read_text())['budgets'].items()}


def window_errors(model, signal: np.ndarray, cond: np.ndarray,
                  window: int, n_windows: int) -> np.ndarray:
    """Reconstruction error per non-overlapping window. Every window is scored — the
    tail short of a full batch is padded out and discarded by ``eval_padded`` — so the
    errors line up 1:1 with the feature/label grid."""
    n_windows = min(n_windows, len(signal) // window, len(cond))
    if n_windows <= 0:
        return np.empty(0, dtype=np.float32)
    windows = (signal[:n_windows * window]
               .reshape(n_windows, window, signal.shape[-1]).astype(np.float32))
    out = eval_padded(model, windows, cond[:n_windows].astype(np.float32))
    return out['error'].reshape(-1).astype(np.float32)

def score_windows(model, signal: np.ndarray, cond: np.ndarray,
                  window: int, n_windows: int) -> dict[str, np.ndarray]:
    recon = window_errors(model, signal, cond, window, n_windows)
    bvp = signal[:, 0]
    spectral = np.empty(len(recon), dtype=np.float32)
    rr = np.empty(len(recon), dtype=np.float32)
    for w in range(len(recon)):
        win = bvp[w * window:(w + 1) * window]
        spectral[w] = dsp.spectral_entropy(win)
        rr[w] = dsp.rr_variability(win)
    return {'recon': recon, 'spectral': spectral, 'rr': rr}


def clean_threshold(clean_score: np.ndarray, budget: float) -> float:
    return float(np.quantile(clean_score, 1.0 - budget))


def subject_thresholds(clean: dict[str, dict[str, np.ndarray]],
                       budgets: dict[str, float]) -> dict[str, dict[str, float]]:
    return {sid: {n: clean_threshold(sc[n], budgets[n]) for n in SCORE_NAMES}
            for sid, sc in clean.items()}


def predict(scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    crossings = [scores[name] > thresholds[name] for name in SCORE_NAMES]
    return np.logical_or.reduce(crossings)


def pooled_predict(scores: dict[str, dict[str, np.ndarray]],
                   thresholds: dict[str, dict[str, float]]) -> np.ndarray:
    return np.concatenate([predict(scores[sid], thresholds[sid]) for sid in scores])


def soft_score(mixed_scores: dict[str, np.ndarray], clean_scores: dict[str, np.ndarray],
               budgets: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(mixed_scores['recon']), dtype=np.float32)
    for n in SCORE_NAMES:
        b = budgets[n]
        if b <= 0.0:
            continue
        clean_sorted = np.sort(clean_scores[n])
        cdf = np.searchsorted(clean_sorted, mixed_scores[n], side='right') / len(clean_sorted)
        out = np.maximum(out, np.clip((cdf - (1.0 - b)) / b, 0.0, 1.0))
    return out


def median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.astype(np.float32, copy=True)
    prev = np.concatenate([x[:1], x[:-1]])
    nxt = np.concatenate([x[1:], x[-1:]])
    return np.median(np.stack([prev, x, nxt]), axis=0).astype(np.float32)


def score_dir_by_subject(trainer: AutoencoderTrainer, data_dir: Path,
                         bvp_dir: Path | None) -> dict[str, dict[str, np.ndarray]]:
    window = trainer.model.seq_len
    subjects_dir = data_dir / CLEAN_SUBDIR
    out: dict[str, dict[str, np.ndarray]] = {}
    for d in get_sorted_paths(subjects_dir):
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=bvp_dir)
        if len(cond) > 0:
            out[sid] = score_windows(trainer.model, signal, cond, window, len(cond))
    return out


def score_mixed_by_subject(trainer: AutoencoderTrainer, data_dir: Path
                           ) -> dict[str, dict[str, np.ndarray]]:
    window = trainer.model.seq_len
    subjects_dir = data_dir / CLEAN_SUBDIR
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = get_sorted_paths(mixed_dir)
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    scores: dict[str, dict[str, np.ndarray]] = {}
    for d in subject_dirs:
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=mixed_dir)
        n_windows = len(np.load(feature_dir / sid / 'features.npy'))
        scores[sid] = score_windows(trainer.model, signal, cond, window, n_windows)
    return scores


def load_mixed_truth(data_dir: Path, scores: dict[str, dict[str, np.ndarray]]
                     ) -> dict[str, np.ndarray]:
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR
    return {sid: np.load(feature_dir / sid / 'labels.npy').reshape(-1)[:len(sc['recon'])]
            for sid, sc in scores.items()}
