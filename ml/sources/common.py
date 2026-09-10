from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class DataSource(ABC):
    """Everything a trainer needs to read one dataset, already through whatever load-time
    filter its DatasetSpec applies and already z-scored — no model normalizes its own
    input. One source is always a single modality and a single variant (clean, mixed, one
    anomaly kind, ...); which modality/variant it serves is fixed by ml.dataset_list's
    registry, not by an argument here, so a new kind of dataset (image, text, ...) only
    has to implement this shape."""

    key: str

    @abstractmethod
    def subject_ids(self) -> list[str]:
        """Subjects this source can serve, in a stable order."""

    @abstractmethod
    def datapoints(self, sid: str) -> np.ndarray:
        """``(n, *shape)`` already-normalized, model-ready array for one subject."""

    def labels(self, sid: str) -> np.ndarray | None:
        """Optional per-datapoint ground truth aligned to ``datapoints()``. None when
        this source carries no ground truth (e.g. a clean-only or single-anomaly-kind
        variant, whose label is implicit in which source it is)."""
        return None

    @abstractmethod
    def calibration_data(self, per_subject: int = 10) -> np.ndarray:
        """A small sample of datapoints across every subject, for int8 calibration."""


def to_dataset(*arrays: np.ndarray) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(tuple(arrays))


def batched(ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    return (ds.shuffle(1000, reshuffle_each_iteration=False)
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
