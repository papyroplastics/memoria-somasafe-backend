"""Fixed spectral descriptor of a BVP window, computed in-graph.

The autoencoder in ``ml.models.spectral_autoencoder`` does not reconstruct the waveform:
it reconstructs this descriptor. Reconstruction error on the raw waveform tracks how
*complex* a window is, so anomalies that simplify the signal (a slowed rhythm above all)
score lower than normal windows and the detector reads them as extra-normal. The
descriptor replaces that with quantities whose normal range is bounded on both sides —
rhythm, spectral shape, waveform regularity — so any departure, faster or slower, leaves
the region the autoencoder learned to reproduce.

Nothing here is trainable: it is a fixed transform in front of the model, the in-graph
counterpart of the C feature extractor the firmware already runs (see
``ml.preprocessing.extract_features``).
"""

import numpy as np
import tensorflow as tf

from .preprocessing import BVP_RATE

EPS = 1e-6

# Pulse band. The dominant frequency, centroid, spread and entropy are read inside it;
# everything outside is summarised by the two out-of-band power ratios.
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


def band_indices(seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(seq_len, 1.0 / BVP_RATE)
    band = np.where((freqs >= BAND_LOW) & (freqs <= BAND_HIGH))[0]
    low = np.where(freqs < BAND_LOW)[0]
    high = np.where(freqs > BAND_HIGH)[0]
    return freqs, band, low, high


def descriptor(signal: tf.Tensor, seq_len: int) -> tf.Tensor:
    """``(batch, seq_len, 1)`` normalized BVP -> ``(batch, N_DESCRIPTORS)``."""
    x = signal[:, :, 0]
    x = x - tf.reduce_mean(x, axis=1, keepdims=True)
    std = tf.math.reduce_std(x, axis=1)

    freqs, band, low, high = band_indices(seq_len)
    hann = tf.constant(np.hanning(seq_len), dtype=tf.float32)
    fft = tf.signal.rfft(x * hann)
    power = tf.math.real(fft) ** 2 + tf.math.imag(fft) ** 2
    total = tf.reduce_sum(power, axis=1) + EPS

    band_power = tf.gather(power, band, axis=1)
    band_freqs = tf.constant(freqs[band], dtype=tf.float32)
    band_total = tf.reduce_sum(band_power, axis=1) + EPS

    centroid = tf.reduce_sum(band_power * band_freqs, axis=1) / band_total
    spread = tf.sqrt(tf.reduce_sum(
        band_power * (band_freqs - centroid[:, None]) ** 2, axis=1) / band_total)
    shape = band_power / band_total[:, None]
    entropy = -tf.reduce_sum(shape * tf.math.log(shape + 1e-12), axis=1)
    dom = tf.gather(band_freqs, tf.argmax(band_power, axis=1))

    # Wiener-Khinchin on the unwindowed signal: the autocorrelation of a zero-padded
    # window is the inverse transform of its power spectrum, so a second forward
    # transform buys both the pulse-period estimate and how regular that period is.
    pad = tf.signal.rfft(x, fft_length=[2 * seq_len])
    ac = tf.signal.irfft(tf.cast(tf.math.real(pad) ** 2 + tf.math.imag(pad) ** 2,
                                 tf.complex64), fft_length=[2 * seq_len])
    ac = ac / (ac[:, :1] + EPS)
    lags = ac[:, LAG_MIN:LAG_MAX]
    ac_lag = tf.cast(tf.argmax(lags, axis=1) + LAG_MIN, tf.float32) / BVP_RATE

    mad = tf.reduce_mean(tf.abs(x[:, 1:] - x[:, :-1]), axis=1)
    moment = lambda k: tf.reduce_mean(x ** k, axis=1) / (std ** k + EPS)

    return tf.stack([
        tf.math.log(std + EPS),
        dom,
        centroid,
        spread,
        entropy,
        tf.math.log(tf.reduce_max(band_power, axis=1) / band_total + EPS),
        tf.math.log(tf.reduce_sum(tf.gather(power, low, axis=1), axis=1) / total + EPS),
        tf.math.log(tf.reduce_sum(tf.gather(power, high, axis=1), axis=1) / total + EPS),
        tf.math.log(band_total / total + EPS),
        tf.math.log(mad / (std + EPS) + EPS),
        tf.reduce_max(lags, axis=1),
        ac_lag,
        moment(3),
        moment(4),
    ], axis=1)
