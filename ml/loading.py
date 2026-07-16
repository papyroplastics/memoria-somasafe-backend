"""tf.data pipelines over the arrays ``ml.preprocessing`` wrote to disk.

Every dataset a training loop, sweep or export path consumes is built here, and
every one of them is cached: windowing a subject is the expensive part of a run and
it never depends on the model weights, so a sweep that rebuilds a fresh-weight
trainer per configuration pays for it once per process.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from .preprocessing import (
    NORM_PARAMS_FILE, FEATURE_STATS_FILE, 
    DatasetUnavailibleError, get_sorted_paths
)

# Keyed by everything that changes the tensors: the source tree, the subject, and the
# windowing/batching applied on top (see Trainer.dataset_key). A dataset carries no model
# state, so two trainers that agree on that key can safely share one.
_subject_cache: dict[tuple, tf.data.Dataset] = {}

def load_norm_params(subjects_dir: Path) -> tuple[float, float]:
    """Global BVP (mean, std) from Stage 1, guarded against a zero std."""
    params = np.load(subjects_dir / NORM_PARAMS_FILE)
    return float(params[0]), float(params[1]) + 1e-8


def norm_stats(subjects_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    bvp_mean, bvp_std = load_norm_params(subjects_dir)
    return np.array([bvp_mean], dtype=np.float32), np.array([bvp_std], dtype=np.float32)


def load_feature_stats(feature_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(feature_dir / FEATURE_STATS_FILE)
    return stats[0].astype(np.float32), stats[1].astype(np.float32)


def load_signal(signal_dir: Path, sid: str) -> np.ndarray:
    """Raw, un-normalized ``(T, 1)`` BVP for a subject, read from ``signal_dir`` —
    clean-signals, mixed-signals or a per-kind anomalous-signals directory."""
    bvp = np.load(signal_dir / sid / 'bvp.npy').astype(np.float32)
    return bvp.reshape(-1, 1)


def window_count(signal: np.ndarray, window_size: int) -> int:
    return max(0, (len(signal) - window_size) // window_size + 1)


def subject_dirs(data_root: Path, subdir: str) -> list[Path]:
    data_dir = data_root / subdir
    dirs = get_sorted_paths(data_dir)
    if not dirs:
        raise DatasetUnavailibleError(data_dir)
    return dirs


def window_signal(signal: np.ndarray, window_size: int, shift: int) -> tf.data.Dataset:
    count = (len(signal) - window_size) // shift + 1
    return (tf.data.Dataset.from_tensor_slices(signal)
            .window(size=window_size, shift=shift, drop_remainder=True)
            .flat_map(lambda w: w.batch(window_size, drop_remainder=True))
            .apply(tf.data.experimental.assert_cardinality(count)))


def subject_windows(signal_dir: Path, sid: str, window_size: int,
                    shift: int) -> tf.data.Dataset:
    ds = window_signal(load_signal(signal_dir, sid), window_size, shift)
    return ds.map(lambda w: (w,))


def cached(key: tuple, build) -> tf.data.Dataset:
    if key not in _subject_cache:
        _subject_cache[key] = build()
    return _subject_cache[key]


def batched(ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    return (ds.shuffle(len(ds), reshuffle_each_iteration=False)
              .batch(batch_size, drop_remainder=True)
              .cache())


def pool(datasets: list[tf.data.Dataset]) -> tf.data.Dataset:
    count = sum(len(ds) for ds in datasets)
    return (tf.data.Dataset
            .sample_from_datasets(datasets, rerandomize_each_iteration=False)
            .apply(tf.data.experimental.assert_cardinality(count))
            .cache())


def holdout(datasets: list[tf.data.Dataset], n_eval: int
            ) -> tuple[list[tf.data.Dataset], list[tf.data.Dataset]]:
    if n_eval < 0:
        raise ValueError(f"n_eval must be >= 0, got {n_eval}")
    if n_eval >= len(datasets):
        raise ValueError(f"n_eval {n_eval} leaves no training subjects "
                         f"({len(datasets)} available)")
    if n_eval == 0:
        return datasets, []
    return datasets[:-n_eval], datasets[-n_eval:]

