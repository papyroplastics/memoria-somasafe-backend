"""tf.data pipelines over the arrays ``ml.preprocessing`` wrote to disk.

Every dataset a training loop, sweep or export path consumes is built here, and
every one of them is cached: windowing a subject is the expensive part of a run and
it never depends on the model weights, so a sweep that rebuilds a fresh-weight
trainer per configuration pays for it once per process.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from .preprocessing import DatasetUnavailibleError, get_sorted_paths, load_signal

# Keyed by everything that changes the tensors: the source tree, the subject, and the
# windowing/batching applied on top (see Trainer.dataset_key). A dataset carries no model
# state, so two trainers that agree on that key can safely share one.
_subject_cache: dict[tuple, tf.data.Dataset] = {}


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
    """One-tuple ``(bvp_window,)`` datapoints for one subject, raw — the model
    z-scores the signal in its own signatures. ACC is not fed to any model; it exists
    only as a feature-extraction input."""
    ds = window_signal(load_signal(signal_dir, sid), window_size, shift)
    return ds.map(lambda w: (w,))


def cached(key: tuple, build) -> tf.data.Dataset:
    if key not in _subject_cache:
        _subject_cache[key] = build()
    return _subject_cache[key]


def batched(ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """Shuffle a subject's datapoints, batch them and hold the result in memory. The
    shuffle is fixed across iterations so the cache is a stable dataset, not a new
    ordering every epoch."""
    return (ds.shuffle(len(ds), reshuffle_each_iteration=False)
              .batch(batch_size, drop_remainder=True)
              .cache())


def pool(datasets: list[tf.data.Dataset]) -> tf.data.Dataset:
    """Merge per-subject datasets into one stream. ``sample_from_datasets`` interleaves
    the subjects randomly and each subject's datapoints are already shuffled within it,
    so the result is not ordered by subject and needs no further shuffling."""
    count = sum(len(ds) for ds in datasets)
    return (tf.data.Dataset
            .sample_from_datasets(datasets, rerandomize_each_iteration=False)
            .apply(tf.data.experimental.assert_cardinality(count))
            .cache())


def holdout(datasets: list[tf.data.Dataset], n_eval: int
            ) -> tuple[list[tf.data.Dataset], list[tf.data.Dataset]]:
    """Split at *subject* granularity: the last ``n_eval`` subjects are held out whole,
    so a metric measured on them is generalization to an unseen subject rather than to
    unseen windows of a subject the model already trained on."""
    if n_eval < 0:
        raise ValueError(f"n_eval must be >= 0, got {n_eval}")
    if n_eval >= len(datasets):
        raise ValueError(f"n_eval {n_eval} leaves no training subjects "
                         f"({len(datasets)} available)")
    if n_eval == 0:
        return datasets, []
    return datasets[:-n_eval], datasets[-n_eval:]

