"""Sliding-window ``tf.data`` builders over the preprocessed DaLiA arrays.

Ported from the legacy ``DaLiADataset`` (PyTorch). Each window yields the tuple
``(signal, context, static)`` where:
  - ``signal``  is a ``(window_size, n_signals)`` slice of [BVP, ACC],
  - ``context`` is the activity context at the window's last timestep ``(2,)``,
  - ``static``  is the per-subject demographics vector ``(6,)``, broadcast to
    every window.

``build_subject_dataset`` returns the full, unbatched, unshuffled dataset.
Callers are responsible for splitting (``take``/``skip``), shuffling, and
batching as needed.

The autoencoder reconstructs ``signal``, so the target is the signal itself and
is not emitted separately; the training loops reuse the input.
"""

import pathlib
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

WINDOW_SIZE = 128
STRIDE = 32


def load_subject(data_dir: Path, subject_id: int):
    subject_dir = data_dir / f"S{subject_id}"
    signal = np.load(subject_dir / 'signal.npy')
    context = np.load(subject_dir / 'context.npy')
    static = np.load(subject_dir / 'static.npy')
    return signal, context, static


def window_arrays(
    signal: np.ndarray,
    context: np.ndarray,
    static: np.ndarray,
    window_size: int,
    stride: int,
):
    n_samples = signal.shape[0]
    num_windows = (n_samples - window_size) // stride + 1
    if num_windows <= 0:
        raise ValueError(
            f"Split too short for a {window_size}-sample window "
            f"(have {n_samples} samples)")

    signals = np.empty((num_windows, window_size, signal.shape[1]), dtype=np.float32)
    contexts = np.empty((num_windows, context.shape[1]), dtype=np.float32)

    for w in range(num_windows):
        start = w * stride
        end = start + window_size
        signals[w] = signal[start:end]
        contexts[w] = context[end - 1]

    statics = np.repeat(static[None, :], num_windows, axis=0).astype(np.float32)
    return signals, contexts, statics


def build_subject_dataset(
    data_dir: Path,
    subject_id: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tf.data.Dataset:
    signal, context, static = load_subject(data_dir, subject_id)
    signals, contexts, statics = window_arrays(
        signal, context, static, window_size, stride)
    return tf.data.Dataset.from_tensor_slices((signals, contexts, statics))
