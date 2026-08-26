"""tf.data plumbing over the arrays a ``ml.sources`` DataSource hands back.

Reading the dataset off disk — layout, windowing, activity filtering — belongs to the
source; this module only turns the resulting numpy arrays into the batched, pooled and
split pipelines the training loops consume.
"""

import numpy as np
import tensorflow as tf


def to_dataset(*arrays: np.ndarray) -> tf.data.Dataset:
    """One subject's datapoints, unbatched: each array contributes one tensor per
    datapoint, in the order ``Trainer.dataset_tensors`` names them."""
    return tf.data.Dataset.from_tensor_slices(tuple(arrays))


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
