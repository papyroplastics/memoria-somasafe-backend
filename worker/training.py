from worker.models import TrainableModel
import tensorflow as tf

@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor):
    return tf.reduce_mean((y-x)**2)

def basic_train_loop(
        model: TrainableModel,
        train_dataset: tf.data.Dataset,
        eval_dataset: tf.data.Dataset,
        epochs: int):

    for epoch in range(epochs):
        train_loss = 0.0

        for batch_x, batch_y  in train_dataset:
            train_loss = model.train(batch_x, batch_y)['loss']

        if epoch % 10 == 0:
            eval_loss = tf.reduce_mean([mse_loss(model.eval(vb_x)['result'], vb_y) for vb_x, vb_y in eval_dataset], 0)

            print(f"epoch={epoch:03d} train loss={train_loss:.6f} eval_loss={eval_loss:.6f}")


def autoencoder_eval(model: TrainableModel, eval_dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.reduce_mean([
        mse_loss(model.eval(signal, context, static)['reconstruction'], signal)
        for (signal, context, static) in eval_dataset
    ], 0)


def dataset_slice(ds: tf.data.Dataset, slice_idx: int, num_slices: int):
    return ds.skip(slice_idx * (len(ds)//num_slices)).take(len(ds)//num_slices) 

def autoencoder_train_loop(
        model: TrainableModel,
        subject_train_datasets: list[tf.data.Dataset],
        eval_dataset: tf.data.Dataset,
        num_slices: int,
        num_passes: int):

    for pass_idx in range(num_passes):
        for slice_idx in range(num_slices):
            combined = tf.data.Dataset.sample_from_datasets([
                dataset_slice(ds, slice_idx, num_slices) for ds in subject_train_datasets
            ])

            print(f"pass={pass_idx + 1}/{num_passes} slice={slice_idx + 1}/{num_slices} ", end="")

            train_loss = 0.0
            for signal, context, static in combined:
                train_loss = model.train(signal, context, static)['loss']
            print(f"train_loss={train_loss:.6f} ", end="")

            eval_loss = autoencoder_eval(model, eval_dataset)
            print(f"eval_loss={eval_loss:.6f}")


def fed_avg(vectors: list[tf.Tensor], sizes: list[int]) -> tf.Tensor:
    total = sum(sizes)
    avg = tf.zeros(vectors[0].shape)

    for vector, size in zip(vectors, sizes):
        avg += vector * (size / total)

    return avg

def federated_train_eval_loop(
        model: TrainableModel,
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

