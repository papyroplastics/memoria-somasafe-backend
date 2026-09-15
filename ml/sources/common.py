from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class DataSource(ABC):
    """One dataset through whatever load-time filter its registry entry applies, as a
    single modality and variant."""

    key: str

    @abstractmethod
    def subject_ids(self) -> list[str]:
        """Subjects this source can serve, in a stable order."""

    @abstractmethod
    def datapoints(self, sid: str) -> np.ndarray:
        """``(n, *shape)`` model-ready array for one subject."""

    def labels(self, sid: str) -> np.ndarray | None:
        """Per-datapoint ground truth aligned to ``datapoints()``, when the source has any."""
        return None

    @abstractmethod
    def calibration_data(self, per_subject: int = 10) -> np.ndarray:
        """A small sample of datapoints across every subject, for int8 calibration."""


def to_dataset(*arrays: np.ndarray) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(tuple(arrays))


def batched(ds: tf.data.Dataset, batch_size: int, shuffle_buffer: int = 1000,
            cache: bool = True) -> tf.data.Dataset:
    ds = (ds.shuffle(shuffle_buffer, reshuffle_each_iteration=False)
            .batch(batch_size, drop_remainder=True))
    return ds.cache() if cache else ds


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
