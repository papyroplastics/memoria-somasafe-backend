from pathlib import Path
import numpy as np
import tensorflow as tf

def build_subject_dataset(
    subject_dir: Path,
    window_size: int,
    shift: int
) -> tf.data.Dataset:
    signal = np.load(subject_dir / 'signal.npy')
    context = np.load(subject_dir / 'context.npy')
    static = np.load(subject_dir / 'static.npy')

    window_count = (len(signal) - window_size) // shift + 1

    signal_ds = (tf.data.Dataset.from_tensor_slices(signal)
        .window(size=window_size, shift=shift, drop_remainder=True)
        .flat_map(lambda w: w.batch(window_size, drop_remainder=True))
        .apply(tf.data.experimental.assert_cardinality(window_count))
    )
    context_ds = tf.data.Dataset.from_tensor_slices(context[::shift][:window_count])

    static_ds = tf.data.Dataset.from_tensor_slices(static)
    static_ds = static_ds.batch(len(static_ds)).repeat()

    return tf.data.Dataset.zip((signal_ds, context_ds, static_ds)) # type: ignore

