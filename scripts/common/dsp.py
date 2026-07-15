"""Rhythm-quality index over a single BVP window."""

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
