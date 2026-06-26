import numpy as np
import tensorflow as tf


@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean((y - x) ** 2)


@tf.function
def first_difference_loss(reconstruction: tf.Tensor, signal: tf.Tensor) -> tf.Tensor:
    """MSE between the temporal first differences of reconstruction and target.

    Penalizes the morphology (slope) of the waveform rather than its level, so a
    constant-output ("flat line") reconstruction is punished even when its mean
    matches the signal. Added alongside ``mse_loss`` in the autoencoder objective."""
    d_recon = reconstruction[:, 1:, :] - reconstruction[:, :-1, :]
    d_signal = signal[:, 1:, :] - signal[:, :-1, :]
    return tf.reduce_mean((d_recon - d_signal) ** 2)


def reconstruction_error(reconstruction: tf.Tensor, signal: tf.Tensor) -> tf.Tensor:
    """Per-window mean squared error — the anomaly score for autoencoders."""
    return tf.reduce_mean(tf.square(reconstruction - signal), axis=[1, 2])


def best_threshold(errors: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Reconstruction-error threshold (predict anomalous when error > threshold)
    that maximizes accuracy against ``labels``. Single sorted sweep."""
    order = np.argsort(errors, kind='stable')
    e = errors[order]
    y = labels[order].astype(bool)
    n = len(y)

    # Split i predicts windows [i, n) anomalous; accuracy = correct negatives in
    # [0, i) + correct positives in [i, n).
    cumneg = np.concatenate([[0], np.cumsum(~y)])          # negatives in [0, i)
    pos_suffix = int(y.sum()) - np.concatenate([[0], np.cumsum(y)])  # positives in [i, n)
    acc = (cumneg + pos_suffix) / n

    i = int(np.argmax(acc))
    if i == 0:
        thr = float(e[0]) - 1.0
    elif i == n:
        thr = float(e[-1]) + 1.0
    else:
        thr = float((e[i - 1] + e[i]) / 2.0)
    return thr, float(acc[i])


def classification_report(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Precision/recall for a boolean prediction against ground truth (anomaly = 1)."""
    pred = pred.astype(bool)
    truth = truth > 0.5
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'precision': precision, 'recall': recall}
