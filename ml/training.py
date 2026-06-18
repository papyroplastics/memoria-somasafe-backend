import tensorflow as tf

from .models.common import Trainer

History = list[tuple[int, float, dict[str, float]]]


def fed_avg(vectors: list[tf.Tensor], sizes: list[int]) -> tf.Tensor:
    total = sum(sizes)
    avg = tf.zeros(vectors[0].shape)

    for vector, size in zip(vectors, sizes):
        avg += vector * (size / total)

    return avg


def _format(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def normal_loop(trainer: Trainer, train_dataset: tf.data.Dataset,
                eval_dataset: tf.data.Dataset, epochs: int) -> History:
    history: History = []
    for epoch in range(epochs):
        loss = trainer.train_epoch(train_dataset)
        metrics = trainer.evaluate(eval_dataset)
        history.append((epoch, loss, metrics))
        print(f"epoch={epoch:03d} loss={loss:.4f} {_format(metrics)}", flush=True)
    return history


def federated_loop(trainer: Trainer, subject_train_datasets: list[tf.data.Dataset],
                   eval_dataset: tf.data.Dataset, local_epochs: int,
                   global_epochs: int, aggregate=fed_avg) -> History:
    """Simulated FedAvg: each round every subject trains locally from the shared
    weights, then ``aggregate`` merges the updates. ``aggregate`` is injectable
    so other strategies (FedProx, weighted median, ...) can be dropped in."""
    model = trainer.model
    sizes = [len(ds) for ds in subject_train_datasets]
    global_weights = model.save()['parameters']

    history: History = []
    for r in range(global_epochs):
        client_weights: list[tf.Tensor] = []
        loss = 0.0
        for train_ds in subject_train_datasets:
            model.restore(tf.constant(global_weights))
            for _ in range(local_epochs):
                loss = trainer.train_epoch(train_ds)
            client_weights.append(model.save()['parameters'])

        global_weights = aggregate(client_weights, sizes)
        model.restore(tf.constant(global_weights))

        metrics = trainer.evaluate(eval_dataset)
        history.append((r, loss, metrics))
        print(f"round={r:03d} {_format(metrics)}", flush=True)
    return history
