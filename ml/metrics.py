import numpy as np
import tensorflow as tf


@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean((y - x) ** 2)


@tf.function
def first_difference_loss(reconstruction: tf.Tensor, signal: tf.Tensor) -> tf.Tensor:
    def diff(x: tf.Tensor) -> tf.Tensor:
        n = x.shape[1] - 1
        return (tf.slice(x, [0, 1, 0], [-1, n, -1])
                - tf.slice(x, [0, 0, 0], [-1, n, -1]))

    return tf.reduce_mean((diff(reconstruction) - diff(signal)) ** 2)


def reconstruction_error(reconstruction: tf.Tensor, signal: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(reconstruction - signal), axis=[1, 2])


def best_threshold(errors: np.ndarray, labels: np.ndarray,
                   objective: str = 'f1') -> tuple[float, float]:
    order = np.argsort(errors, kind='stable')
    e = errors[order]
    y = labels[order].astype(bool)
    n = len(y)
    pos = int(y.sum())

    # Split i predicts windows [i, n) anomalous (i = 0..n).
    cumneg = np.concatenate([[0], np.cumsum(~y)])           # negatives in [0, i)
    tp = pos - np.concatenate([[0], np.cumsum(y)])          # true positives in [i, n)
    pred_pos = n - np.arange(n + 1)                         # predicted-positive count (tp + fp)

    if objective == 'accuracy':
        score = (cumneg + tp) / n
    elif objective == 'f1':
        denom = pred_pos + pos                              # (tp + fp) + (tp + fn)
        score = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    else:
        raise ValueError(f"unknown objective {objective!r} (expected 'f1' or 'accuracy')")

    i = int(np.argmax(score))
    if i == 0:
        thr = float(e[0]) - 1.0
    elif i == n:
        thr = float(e[-1]) + 1.0
    else:
        thr = float((e[i - 1] + e[i]) / 2.0)
    return thr, float(score[i])


def classification_report(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    truth = truth > 0.5
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
