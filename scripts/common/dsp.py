"""Rhythm-quality indices over a single BVP window. """

import numpy as np

from ml.preprocessing import BVP_RATE

# Plausible heart-rate band (~42–210 bpm); matches the spectral feature band in
# ml.preprocessing.
PHYS_LO_HZ = 0.7
PHYS_HI_HZ = 3.5


def _band_power(window: np.ndarray, rate: int) -> np.ndarray:
    """In-band power spectrum of a Hann-tapered, mean-centred window."""
    x = (window - window.mean()) * np.hanning(len(window))
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0 / rate)
    return power[(freqs >= PHYS_LO_HZ) & (freqs <= PHYS_HI_HZ)]


def spectral_entropy(window: np.ndarray, rate: int = BVP_RATE) -> float:
    """Normalized Shannon entropy of the in-band power spectrum"""
    p = _band_power(window, rate)
    total = p.sum()
    if total <= 0.0 or len(p) < 2:
        return 0.0
    n_bins = len(p)
    p = p[p > 0.0] / total
    return float(-np.sum(p * np.log(p)) / np.log(n_bins))


def _find_peaks(x: np.ndarray, min_distance: int, height: float) -> np.ndarray:
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
    """Coefficient of variation of inter-beat (peak-to-peak) intervals"""
    x = window - window.mean()
    peaks = _find_peaks(x, min_distance=int(rate * 0.3), height=0.5 * x.std())
    if len(peaks) < 3:
        return 0.0
    rr = np.diff(peaks).astype(np.float64)
    mean = rr.mean()
    return float(rr.std() / mean) if mean > 0 else 0.0
