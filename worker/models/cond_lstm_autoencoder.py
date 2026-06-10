import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .common import Dense, LSTMCell, TrainableModel
from ..optimizers import Adam
from ..training import mse_loss, fed_avg
from ..saving import save_tainable_model, save_optimized_model


class ConditionalLSTMAutoencoder(TrainableModel):
    """LSTM autoencoder for unsupervised cardiovascular anomaly detection.

    Reconstructs a window of ``[BVP, ACC]`` and conditions the latent on a
    static demographics vector plus an activity-context vector. The
    reconstruction error is the anomaly score.

    Exposes the same ``eval``/``train``/``save``/``restore`` signatures as
    ``BasicNN`` so it stays LiteRT-trainable on-device and FedAvg can move
    flattened weights. The optimizer is Adam whose state is kept in
    non-trainable variables (excluded from save/restore).
    """

    def __init__(self, name: str, batch_size: int, seq_len: int,
                 n_signals: int, n_static: int, n_context: int,
                 hidden_dim: int, latent_dim: int, cond_embed_dim: int,
                 learning_rate: float, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-7):
        super().__init__(name=name)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_signals = n_signals

        self.signal_shape = (batch_size, seq_len, n_signals)
        self.context_shape = (batch_size, n_context)
        self.static_shape = (batch_size, n_static)

        self.cond_dense1 = Dense(n_static + n_context, 32, activation=tf.nn.relu)
        self.cond_dense2 = Dense(32, cond_embed_dim, activation=tf.nn.relu)

        self.hidden_dim = hidden_dim
        self.encoder_lstm = LSTMCell(n_signals, hidden_dim)
        self.decoder_lstm = LSTMCell(hidden_dim, hidden_dim)

        self.fusion = Dense(hidden_dim + cond_embed_dim, latent_dim, activation=tf.nn.relu)
        self.latent_to_hidden = Dense(latent_dim, hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(hidden_dim, n_signals, activation=None)

        self.optimizer = Adam(
            self.trainable_variables, learning_rate, beta1, beta2, epsilon)

        self.eval = tf.function(self.eval_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.context_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.static_shape, dtype=tf.float32),
        ])

        self.train = tf.function(self.train_eager, input_signature=[
            tf.TensorSpec(shape=self.signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.context_shape, dtype=tf.float32),
            tf.TensorSpec(shape=self.static_shape, dtype=tf.float32),
        ])

        self._init_save_restore()


    def _forward(self, signal, context, static):
        cond = self.cond_dense2(self.cond_dense1(tf.concat([context, static], axis=1)))

        h, c = self.encoder_lstm.zero_state(self.batch_size)
        for t in range(self.seq_len):
            h, c = self.encoder_lstm.step(h, c, signal[:, t, :])

        dec_hidden = self.latent_to_hidden(self.fusion(tf.concat([h, cond], axis=1)))

        dh, dc = self.decoder_lstm.zero_state(self.batch_size)
        outputs = []
        for _ in range(self.seq_len):
            dh, dc = self.decoder_lstm.step(dh, dc, dec_hidden)
            outputs.append(self.output_layer(dh))

        return tf.stack(outputs, axis=1)

    def eval_eager(self, signal: tf.Tensor, context: tf.Tensor, static: tf.Tensor):
        reconstruction = self._forward(signal, context, static)
        error = tf.reduce_mean(tf.square(reconstruction - signal), axis=[1, 2])
        return {'reconstruction': reconstruction, 'error': error}

    def train_eager(self, signal: tf.Tensor, context: tf.Tensor, static: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = mse_loss(self._forward(signal, context, static), signal)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(self.trainable_variables, grads)
        return {'loss': loss}


def autoencoder_eval(model, eval_dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.reduce_mean([
        mse_loss(model.eval(signal, context, static)['reconstruction'], signal)
        for (signal, context, static) in eval_dataset
    ], 0)


def dataset_slice(ds: tf.data.Dataset, num_slices: int, slice_idx: int):
    return ds.skip(slice_idx * (len(ds) // num_slices)).take(len(ds) // num_slices)


def train_loop(
        model,
        subject_train_datasets: list[tf.data.Dataset],
        eval_dataset: tf.data.Dataset,
        num_slices: int,
        num_passes: int):

    for pass_idx in range(num_passes):
        for slice_idx in range(num_slices):
            print(f"pass={pass_idx + 1}/{num_passes} ", end="", flush=True)

            combined = tf.data.Dataset.sample_from_datasets([
                dataset_slice(ds, num_slices, slice_idx) for ds in subject_train_datasets
            ])
            print(f"slice={slice_idx + 1}/{num_slices} ", end="", flush=True)

            train_loss = 0.0
            for signal, context, static in combined:
                train_loss = model.train(signal, context, static)['loss']
            print(f"train_loss={train_loss:.6f} ", end="", flush=True)

            eval_loss = autoencoder_eval(model, eval_dataset)
            print(f"eval_loss={eval_loss:.6f}", flush=True)


def federated_train_eval_loop(
        model,
        subject_train_datasets: list[tf.data.Dataset],
        subject_eval_datasets: list[tf.data.Dataset],
        local_epochs: int,
        global_epochs: int):

    num_subjects = len(subject_train_datasets)
    print(f"Starting federated training over {num_subjects} subjects...")

    subject_sizes = [len(ds) for ds in subject_train_datasets]
    global_weights = model.save()['parameters']

    for r in range(1, global_epochs + 1):
        print(f"\n--- Round {r}/{global_epochs} ---")

        trained_param_list: list[tf.Tensor] = []

        for cid, train_ds in enumerate(subject_train_datasets):
            model.restore(tf.constant(global_weights))
            print(f"    subject {cid + 1}: local losses:", end='')

            for _ in range(local_epochs):
                epoch_loss = 0.0
                for signal, context, static in train_ds:
                    epoch_loss += model.train(signal, context, static)['loss'] / len(train_ds)
                print(f" {epoch_loss:.6f}", end='')

            print()
            trained_param_list.append(model.save()['parameters'])

        global_weights = fed_avg(trained_param_list, subject_sizes)
        model.restore(tf.constant(global_weights))

        eval_loss = tf.reduce_mean([
            autoencoder_eval(model, ds) for ds in subject_eval_datasets
        ])
        print(f"\n    global eval loss: {eval_loss:.6f}\n")

    model.restore(tf.constant(global_weights))


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


def run(data_root: Path, result_dir: Path, seed: int):
    """Reconstruction-based anomaly model on PPG-DaLiA. Exports a trainable
    SavedModel + TFLite; int8 export may fail (LSTM is poorly supported on
    TFLM) and is skipped if so."""
    tf.random.set_seed(seed)

    data_dir = data_root / 'ppg-dalia-processed' / 'subjects'
    if not data_dir.is_dir():
        raise SystemExit(f"Processed dataset not found at {data_dir}. Run get-dataset.py first.")

    sample_rate = 64                    # hz
    window_size = sample_rate * 8       # 8 s windows
    shift = sample_rate * 3             # 3 s stride
    train_split, batch_size, num_slices, num_passes = 0.975, 12, 10, 1

    subject_train_datasets, subject_eval_datasets = [], []
    print("Processed:", end="")
    for d in sorted(data_dir.glob('S*')):
        ds = build_subject_dataset(d, window_size, shift)
        ds = ds.shuffle(len(ds)).batch(batch_size, drop_remainder=True)

        train_count = int(len(ds) * train_split)
        subject_train_datasets.append(ds.take(train_count))
        subject_eval_datasets.append(ds.skip(train_count))
        print(f" {d.name}", end="", flush=True)
    print()

    eval_dataset = tf.data.Dataset.sample_from_datasets(subject_eval_datasets)
    del subject_eval_datasets

    model = ConditionalLSTMAutoencoder(
        name='dalia_lstm_ae', batch_size=batch_size, seq_len=window_size,
        n_signals=2, n_static=6, n_context=2,
        hidden_dim=64, latent_dim=32, cond_embed_dim=16, learning_rate=1e-3,
    )

    train_loop(model, subject_train_datasets, eval_dataset, num_slices, num_passes)

    print("Compiling and saving model")
    saved_model, sm_path = save_tainable_model(result_dir, 'pre-train', model)
    rep_dataset = eval_dataset.map(lambda s, c, st: {'signal': s, 'context': c, 'static': st})
    save_optimized_model(result_dir, 'pre-train', model, rep_dataset)
    print(f"Saved model to {sm_path}")

    for signal, context, static in eval_dataset.take(1):
        recon = saved_model.eval(signal, context, static)['reconstruction']
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(signal[0].numpy())
        axs[0].set_title('Input window [BVP, ACC]')
        axs[1].plot(recon[0].numpy())
        axs[1].set_title('Reconstruction')
        fig.savefig(result_dir / 'reconstruction.png')
        print(f"saved reconstruction plot to {result_dir / 'reconstruction.png'}")
        break
