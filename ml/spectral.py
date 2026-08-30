"""Fixed spectral descriptor of a BVP window, reconstructed by
``ml.models.spectral_autoencoder`` in place of the raw waveform so anomalies that
merely simplify the signal don't score as extra-normal. Plain numpy, not trainable."""

import numpy as np

from .preprocessing import BVP_RATE

EPS = 1e-9

# Pulse band. The dominant frequency, centroid, spread and entropy are read inside it;
# everything outside is summarized by the two out-of-band power ratios.
BAND_LOW = 0.5
BAND_HIGH = 4.0

# Autocorrelation lag range, in samples: one pulse period at 3 Hz through one at 0.6 Hz.
LAG_MIN = int(BVP_RATE / 3.0)
LAG_MAX = int(BVP_RATE / 0.6)

DESCRIPTOR_NAMES = (
    'log_std', 'dom_freq', 'centroid', 'spread', 'entropy', 'peak_frac',
    'low_ratio', 'high_ratio', 'band_ratio', 'mad_ratio', 'ac_max', 'ac_lag',
    'skew', 'kurtosis',
)
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def descriptors(windows: np.ndarray) -> np.ndarray:
    """``(n, window)`` or ``(n, window, 1)`` raw BVP -> ``(n, N_DESCRIPTORS)`` float32."""
    x = windows.reshape(len(windows), -1).astype(np.float64)
    n_samples = x.shape[1]
    if not len(x):
        return np.empty((0, N_DESCRIPTORS), dtype=np.float32)

    x = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1)

    freqs = np.fft.rfftfreq(n_samples, 1.0 / BVP_RATE)
    power = np.abs(np.fft.rfft(x * np.hanning(n_samples), axis=1)) ** 2
    total = power.sum(axis=1) + EPS

    band = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)
    band_power, band_freqs = power[:, band], freqs[band]
    band_total = band_power.sum(axis=1) + EPS

    centroid = (band_power * band_freqs).sum(axis=1) / band_total
    spread = np.sqrt((band_power * (band_freqs - centroid[:, None]) ** 2).sum(axis=1)
                     / band_total)
    shape = band_power / band_total[:, None]

    # Wiener-Khinchin on the unwindowed signal: the autocorrelation of a zero-padded
    # window is the inverse transform of its power spectrum, so one more transform buys
    # both the pulse-period estimate and how regular that period is.
    padded = np.fft.rfft(x, n=2 * n_samples, axis=1)
    autocorr = np.fft.irfft(padded * np.conj(padded), axis=1)[:, :n_samples]
    autocorr = autocorr / (autocorr[:, :1] + EPS)
    lags = autocorr[:, LAG_MIN:LAG_MAX]

    mad = np.abs(np.diff(x, axis=1)).mean(axis=1)
    moment = lambda k: (x ** k).mean(axis=1) / (std ** k + EPS)

    return np.stack([
        np.log(std + EPS),
        band_freqs[np.argmax(band_power, axis=1)],
        centroid,
        spread,
        -(shape * np.log(shape + 1e-12)).sum(axis=1),
        np.log(band_power.max(axis=1) / band_total + EPS),
        np.log(power[:, freqs < BAND_LOW].sum(axis=1) / total + EPS),
        np.log(power[:, freqs > BAND_HIGH].sum(axis=1) / total + EPS),
        np.log(band_total / total + EPS),
        np.log(mad / (std + EPS) + EPS),
        lags.max(axis=1),
        (np.argmax(lags, axis=1) + LAG_MIN) / BVP_RATE,
        moment(3),
        moment(4),
    ], axis=1).astype(np.float32)
