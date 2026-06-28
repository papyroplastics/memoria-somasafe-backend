"""Rhythm-quality indices over a single BVP window.

These are the baseline-free signal-quality scores OR'd alongside the autoencoder's
reconstruction error in the label-distillation / evaluation pipeline. Reconstruction
MSE is an amplitude/integrity detector (spike/blowup/noise) but is phase- and
rate-insensitive on a quasi-periodic pulse, so it is weak on rhythm anomalies. These
indices target the irregularly-irregular case (afib): a clean pulse concentrates its
energy in a sharp fundamental + harmonics and beats at near-constant intervals, while
an irregular rhythm smears the spectrum and scatters the beat-to-beat intervals.

Both are oriented *higher = more anomalous*, scale/rate invariant (so a fast-but-regular
pulse — timewarp — stays low; that case needs the activity-expected-HR check), and use
only FFT + local-maxima picking so they port to jDSP on-device. Operate on the
normalized BVP channel; the affine load-time normalization leaves spectra and peak
positions unchanged.
"""

import numpy as np

from ml.data import BVP_RATE

# Plausible heart-rate band (~42–210 bpm); matches the spectral feature band in ml.data.
PHYS_LO_HZ = 0.7
PHYS_HI_HZ = 3.5


def _band_power(window: np.ndarray, rate: int) -> np.ndarray:
    """In-band power spectrum of a Hann-tapered, mean-centred window."""
    x = (window - window.mean()) * np.hanning(len(window))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0 / rate)
    return power[(freqs >= PHYS_LO_HZ) & (freqs <= PHYS_HI_HZ)]


def spectral_entropy(window: np.ndarray, rate: int = BVP_RATE) -> float:
    """Normalized Shannon entropy of the in-band power spectrum (0 = all energy in one
    bin, 1 = flat/white). Low for a periodic pulse (energy in a few sharp peaks), high
    for an irregular rhythm or broadband noise. Higher = more anomalous."""
    p = _band_power(window, rate)
    total = p.sum()
    if total <= 0.0 or len(p) < 2:
        return 0.0
    n_bins = len(p)
    p = p[p > 0.0] / total
    return float(-np.sum(p * np.log(p)) / np.log(n_bins))


def _find_peaks(x: np.ndarray, min_distance: int, height: float) -> np.ndarray:
    """Indices of local maxima above ``height``, greedily suppressing any weaker peak
    within ``min_distance`` samples of a stronger one (a beat refractory period)."""
    if len(x) < 3:
        return np.empty(0, dtype=int)
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    cand = cand[x[cand] > height]
    taken = np.zeros(len(x), dtype=bool)
    keep = []
    for idx in cand[np.argsort(x[cand])[::-1]]:
        if not taken[max(0, idx - min_distance):idx + min_distance + 1].any():
            keep.append(idx)
            taken[idx] = True
    return np.sort(np.asarray(keep, dtype=int))


def rr_variability(window: np.ndarray, rate: int = BVP_RATE) -> float:
    """Coefficient of variation of inter-beat (peak-to-peak) intervals. Rate-invariant,
    so a regular pulse stays low at any tempo while an irregularly-irregular rhythm
    (afib) is high. Returns 0 when too few beats are detected. Higher = more anomalous."""
    x = window - window.mean()
    peaks = _find_peaks(x, min_distance=int(rate * 0.3), height=0.5 * x.std())
    if len(peaks) < 3:
        return 0.0
    rr = np.diff(peaks).astype(np.float64)
    mean = rr.mean()
    return float(rr.std() / mean) if mean > 0 else 0.0
