"""Per-window anomaly scoring shared by autoencoder_test.py and distill_labels.py.

The detector is an OR of independently-thresholded scores, each oriented
*higher = more anomalous*:

  - ``recon``    reconstruction MSE from the autoencoder — amplitude/integrity
                 anomalies (spike, blowup, noise).
  - ``spectral`` in-band spectral entropy — irregular/smeared rhythm (afib), noise.
  - ``rr``       beat-interval coefficient of variation — irregular rhythm (afib).

Each score gets its own threshold (picked on the mixed set by autoencoder_test.py);
a window is anomalous if *any* score crosses its threshold. The rhythm scores cover
the rhythm anomalies reconstruction error is structurally blind to; uniform-tempo
timewarp is regular and stays below all three (handled by a later activity-expected-HR
check, see the roadmap).
"""

import numpy as np

from . import dsp
from .autoencoders import window_errors

SCORE_NAMES = ('recon', 'spectral', 'rr')


def score_windows(model, signal: np.ndarray, cond: np.ndarray,
                  window: int, n_windows: int) -> dict[str, np.ndarray]:
    """All per-window scores for one subject. The reconstruction step decides the
    window count (it may drop a batch remainder); the rhythm scores are computed for
    exactly that many windows so every score array lines up."""
    recon = window_errors(model, signal, cond, window, n_windows)
    bvp = signal[:, 0]
    spectral = np.empty(len(recon), dtype=np.float32)
    rr = np.empty(len(recon), dtype=np.float32)
    for w in range(len(recon)):
        win = bvp[w * window:(w + 1) * window]
        spectral[w] = dsp.spectral_entropy(win)
        rr[w] = dsp.rr_variability(win)
    return {'recon': recon, 'spectral': spectral, 'rr': rr}


def pick_thresholds(clean_scores: dict[str, np.ndarray],
                    target_fpr: float = 0.02) -> dict[str, float]:
    """Threshold each score at the ``1 - target_fpr`` quantile of its CLEAN-window
    distribution, so each score fires on ~``target_fpr`` of clean windows.

    Picked off the clean distribution rather than by maximizing F1 against balanced
    labels: an F1 sweep collapses to 'flag everything' whenever a score separates only
    weakly (all-positive scores F1≈0.67 on a 50/50 set, which a soft score can't beat),
    whereas a clean-quantile cut can't go degenerate and directly bounds the OR-combined
    false-alarm rate at roughly ``len(SCORE_NAMES) * target_fpr``."""
    return {name: float(np.quantile(clean_scores[name], 1.0 - target_fpr))
            for name in SCORE_NAMES}


def predict(scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    """OR each score's threshold crossing into one boolean anomaly flag per window."""
    crossings = [scores[name] > thresholds[name] for name in SCORE_NAMES]
    return np.logical_or.reduce(crossings)
