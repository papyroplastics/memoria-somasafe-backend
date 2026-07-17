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
from ml.preprocessing import CLEAN_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR, get_sorted_paths
from ml.loading import load_signal, window_count, holdout
from ml.metrics import classification_report
from ml.models.common import AutoencoderTrainer

DETECTOR = 'recon'       # the autoencoder score the detector and the labels are built on

CALIBRATION_REPORT = 'calibration.json'   # the full FPR sweep, from calibrate_fpr

# Candidate expected FPRs. Because each threshold is the (1 - f) quantile of the subject's
# *own* clean scores, that fraction of clean windows lies above it by definition — so
# J(f) = recall(f) - f and the grid just has to be wide enough to bracket the turn.
FPR_GRID = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
            0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


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
    return {DETECTOR: window_errors(model, signal, window, n_windows)}


def clean_threshold(clean_score: np.ndarray, expected_fpr: float) -> float:
    return float(np.quantile(clean_score, 1.0 - expected_fpr))


def subject_thresholds(clean: dict[str, dict[str, np.ndarray]],
                       expected_fpr: float) -> dict[str, dict[str, float]]:
    """Each subject's detector threshold at the shared global expected FPR."""
    return {sid: {DETECTOR: clean_threshold(sc[DETECTOR], expected_fpr)}
            for sid, sc in clean.items()}


def calibrate_expected_fpr(clean: dict[str, dict[str, np.ndarray]],
                           mixed: dict[str, dict[str, np.ndarray]],
                           truth: dict[str, np.ndarray],
                           grid=FPR_GRID) -> tuple[float, list[dict]]:
    """The expected FPR maximizing the detector's Youden's J, plus the whole sweep.

    J rather than F1 because the mixed set is ~50% anomalous by construction: F1 depends on
    that prevalence, while J = recall - FPR is prevalence-independent. F1 and precision are
    recorded per level anyway, for the report table."""
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])
    sweep = []
    for f in grid:
        thr = {sid: clean_threshold(clean[sid][DETECTOR], f) for sid in clean}
        rep = classification_report(
            np.concatenate([mixed[sid][DETECTOR] > thr[sid] for sid in mixed]), pooled_truth)
        fpr = float(np.concatenate(
            [clean[sid][DETECTOR] > thr[sid] for sid in clean]).mean())
        sweep.append({'expected_fpr': f, 'recall': rep['recall'], 'precision': rep['precision'],
                      'f1': rep['f1'], 'clean_fpr': fpr, 'youden_j': rep['recall'] - fpr})
    best = max(sweep, key=lambda row: row['youden_j'])
    return best['expected_fpr'], sweep


def pooled_flags(scores: dict[str, dict[str, np.ndarray]],
                 thresholds: dict[str, dict[str, float]],
                 name: str = DETECTOR) -> np.ndarray:
    """Per-window fire/no-fire for one score, pooled over subjects in ``scores`` order —
    each subject against its own threshold."""
    return np.concatenate([scores[sid][name] > thresholds[sid][name] for sid in scores])


def split_subject_ids(data_dir: Path, n_eval: int) -> tuple[list[str], list[str]]:
    """(training, held-out) subject IDs for the same last-N split train.py's holdout makes:
    subjects numerically sorted, the last n_eval held out."""
    sids = [d.name for d in get_sorted_paths(data_dir / CLEAN_SUBDIR)]
    return holdout(sids, n_eval)


def score_dir_by_subject(trainer: AutoencoderTrainer, data_dir: Path, bvp_dir: Path | None,
                         subjects: set[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Score every subject's non-overlapping windows. ``bvp_dir`` is the signal source —
    a per-kind anomalous-signals directory, or ``None`` for the clean signals. ``subjects``
    restricts scoring to a subset (e.g. the held-out eval subjects)."""
    window = trainer.model.seq_len
    signal_dir = bvp_dir if bvp_dir is not None else data_dir / CLEAN_SUBDIR
    out: dict[str, dict[str, np.ndarray]] = {}
    for d in get_sorted_paths(data_dir / CLEAN_SUBDIR):
        sid = d.name
        if subjects is not None and sid not in subjects:
            continue
        signal = load_signal(signal_dir, sid)
        count = window_count(signal, window)
        if count > 0:
            out[sid] = score_windows(trainer.model, signal, window, count)
    return out


def score_mixed_by_subject(trainer: AutoencoderTrainer, data_dir: Path,
                           subjects: set[str] | None = None
                           ) -> dict[str, dict[str, np.ndarray]]:
    """The mixed set, scored on the feature grid: the window count comes from
    ``mixed-features`` so the scores line up 1:1 with the ground-truth labels. ``subjects``
    restricts scoring to a subset (e.g. the held-out eval subjects)."""
    window = trainer.model.seq_len
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = get_sorted_paths(mixed_dir)
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    scores: dict[str, dict[str, np.ndarray]] = {}
    for d in subject_dirs:
        sid = d.name
        if subjects is not None and sid not in subjects:
            continue
        signal = load_signal(mixed_dir, sid)
        n_windows = len(np.load(feature_dir / sid / 'features.npy'))
        scores[sid] = score_windows(trainer.model, signal, window, n_windows)
    return scores


def load_mixed_truth(data_dir: Path, scores: dict[str, dict[str, np.ndarray]]
                     ) -> dict[str, np.ndarray]:
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR
    return {sid: np.load(feature_dir / sid / 'labels.npy').reshape(-1)[:len(sc[DETECTOR])]
            for sid, sc in scores.items()}
