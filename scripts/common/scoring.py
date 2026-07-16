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

``spectral`` (in-band spectral entropy) is a classical DSP index carried alongside as a
*baseline*, thresholded the same way so its precision/recall are directly comparable.
It is not part of the detector and never reaches the distilled labels: distill_eval.py
reports it so the learned teacher can be read against a hand-crafted one.

See ../../../shared/docs/anomalies-and-distillation.md for why the expected FPR is
calibrated on Youden's J.
"""

import json
from pathlib import Path

import numpy as np

import tensorflow as tf
from ml.preprocessing import (
    CLEAN_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR,
    get_sorted_paths, load_signal, window_count,
)
from ml.models.common import AutoencoderTrainer

from . import dsp
from .reports import get_report_dir

DETECTOR = 'recon'       # the autoencoder score the detector and the labels are built on
BASELINE = 'spectral'    # hand-crafted DSP index, reported by distill_eval as a comparison
SCORE_NAMES = (DETECTOR, BASELINE)

CALIBRATION_REPORT = 'distill_calibration.json'   # the expected FPR, from distill_calibrate


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


def load_expected_fpr(model_name: str) -> float:
    """The global expected FPR picked by distill_calibrate.py."""
    report_path = get_report_dir(model_name) / CALIBRATION_REPORT
    if not report_path.exists():
        raise SystemExit(
            f"no calibration report at {report_path}. Run distill_calibrate '{model_name}' "
            f"first to pick the expected FPR.")
    return float(json.loads(report_path.read_text())['expected_fpr'])


def window_errors(model, signal: np.ndarray, window: int, n_windows: int) -> np.ndarray:
    """Reconstruction error per non-overlapping window. Every window is scored — the
    tail short of a full batch is padded out and discarded by ``eval_padded`` — so the
    errors line up 1:1 with the feature/label grid."""
    n_windows = min(n_windows, len(signal) // window)
    if n_windows <= 0:
        return np.empty(0, dtype=np.float32)
    windows = (signal[:n_windows * window]
               .reshape(n_windows, window, model.n_signals).astype(np.float32))
    out = eval_padded(model, windows)
    return out['error'].reshape(-1).astype(np.float32)


def score_windows(model, signal: np.ndarray, window: int,
                  n_windows: int) -> dict[str, np.ndarray]:
    recon = window_errors(model, signal, window, n_windows)
    bvp = signal[:, 0]
    spectral = np.array([dsp.spectral_entropy(bvp[w * window:(w + 1) * window])
                         for w in range(len(recon))], dtype=np.float32)
    return {DETECTOR: recon, BASELINE: spectral}


def clean_threshold(clean_score: np.ndarray, expected_fpr: float) -> float:
    return float(np.quantile(clean_score, 1.0 - expected_fpr))


def subject_thresholds(clean: dict[str, dict[str, np.ndarray]],
                       expected_fpr: float) -> dict[str, dict[str, float]]:
    """Each subject's per-score threshold at the shared global expected FPR."""
    return {sid: {n: clean_threshold(sc[n], expected_fpr) for n in SCORE_NAMES}
            for sid, sc in clean.items()}


def pooled_flags(scores: dict[str, dict[str, np.ndarray]],
                 thresholds: dict[str, dict[str, float]],
                 name: str = DETECTOR) -> np.ndarray:
    """Per-window fire/no-fire for one score, pooled over subjects in ``scores`` order —
    each subject against its own threshold."""
    return np.concatenate([scores[sid][name] > thresholds[sid][name] for sid in scores])


def soft_score(mixed_scores: dict[str, np.ndarray], clean_scores: dict[str, np.ndarray],
               expected_fpr: float) -> np.ndarray:
    """The teacher's soft [0,1] label: how far past its threshold the detector score
    ranks in the subject's own clean CDF, so ``label > 0`` reproduces the hard decision."""
    n = len(mixed_scores[DETECTOR])
    if expected_fpr <= 0.0:
        return np.zeros(n, dtype=np.float32)
    clean_sorted = np.sort(clean_scores[DETECTOR])
    cdf = np.searchsorted(clean_sorted, mixed_scores[DETECTOR], side='right') / len(clean_sorted)
    return np.clip((cdf - (1.0 - expected_fpr)) / expected_fpr, 0.0, 1.0).astype(np.float32)


def median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.astype(np.float32, copy=True)
    prev = np.concatenate([x[:1], x[:-1]])
    nxt = np.concatenate([x[1:], x[-1:]])
    return np.median(np.stack([prev, x, nxt]), axis=0).astype(np.float32)


def score_dir_by_subject(trainer: AutoencoderTrainer, data_dir: Path,
                         bvp_dir: Path | None) -> dict[str, dict[str, np.ndarray]]:
    """Score every subject's non-overlapping windows. ``bvp_dir`` is the signal source —
    a per-kind anomalous-signals directory, or ``None`` for the clean signals."""
    window = trainer.model.seq_len
    signal_dir = bvp_dir if bvp_dir is not None else data_dir / CLEAN_SUBDIR
    out: dict[str, dict[str, np.ndarray]] = {}
    for d in get_sorted_paths(data_dir / CLEAN_SUBDIR):
        sid = d.name
        signal = load_signal(signal_dir, sid)
        count = window_count(signal, window)
        if count > 0:
            out[sid] = score_windows(trainer.model, signal, window, count)
    return out


def score_mixed_by_subject(trainer: AutoencoderTrainer, data_dir: Path
                           ) -> dict[str, dict[str, np.ndarray]]:
    """The mixed set, scored on the feature grid: the window count comes from
    ``mixed-features`` so the scores line up 1:1 with the ground-truth labels."""
    window = trainer.model.seq_len
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = get_sorted_paths(mixed_dir)
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    scores: dict[str, dict[str, np.ndarray]] = {}
    for d in subject_dirs:
        sid = d.name
        signal = load_signal(mixed_dir, sid)
        n_windows = len(np.load(feature_dir / sid / 'features.npy'))
        scores[sid] = score_windows(trainer.model, signal, window, n_windows)
    return scores


def load_mixed_truth(data_dir: Path, scores: dict[str, dict[str, np.ndarray]]
                     ) -> dict[str, np.ndarray]:
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR
    return {sid: np.load(feature_dir / sid / 'labels.npy').reshape(-1)[:len(sc[DETECTOR])]
            for sid, sc in scores.items()}
