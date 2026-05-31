import numpy as np
import tensorflow as tf
from typing import Callable

@tf.function
def mse_loss(x: tf.Tensor, y: tf.Tensor):
    return tf.reduce_mean((y-x)**2)

def train_loop(
        model: Callable[[tf.Tensor], dict],
        train_f: Callable[[tf.Tensor, tf.Tensor], dict],
        train_dataset: tf.data.Dataset,
        eval_dataset: tf.data.Dataset,
        epochs: int):

    for epoch in range(epochs):
        train_loss = 0.0

        for batch_x, batch_y  in train_dataset:
            train_loss = train_f(batch_x, batch_y)['loss']

        if epoch % 10 == 0:
            eval_loss = tf.reduce_mean([mse_loss(model(vb_x)['result'], vb_y) for vb_x, vb_y in eval_dataset], 0)

            print(f"epoch={epoch:03d} train loss={train_loss:.6f} eval_loss={eval_loss:.6f}")


# --- Reconstruction-autoencoder training (DaLiA) ---------------------------------

def reconstruction_eval(model, dataset: tf.data.Dataset) -> float:
    """Mean reconstruction MSE of ``model`` over ``dataset`` (signal as target)."""
    losses = []
    for signal, context, static in dataset:
        recon = model.eval(signal, context, static)['reconstruction']
        losses.append(mse_loss(recon, signal))
    if not losses:
        return float('nan')
    return float(tf.reduce_mean(losses))


def reconstruction_train_loop(
        model,
        train_dataset: tf.data.Dataset,
        eval_dataset: tf.data.Dataset,
        epochs: int,
        log_every: int = 10) -> float:
    """Centralized training of the conditional autoencoder on one dataset."""
    train_loss = float('nan')
    for epoch in range(epochs):
        for signal, context, static in train_dataset:
            train_loss = float(model.train(signal, context, static)['loss'])

        if epoch % log_every == 0:
            eval_loss = reconstruction_eval(model, eval_dataset)
            print(f"epoch={epoch:03d} train loss={train_loss:.6f} eval_loss={eval_loss:.6f}")
    return train_loss


# --- Federated (FedAvg) simulation ------------------------------------------------

def _count_batches(dataset: tf.data.Dataset) -> int:
    return sum(1 for _ in dataset)


def fed_avg(vectors: list[np.ndarray], sizes: list[int]) -> np.ndarray:
    """Sample-size-weighted average of flattened weight vectors."""
    total = float(sum(sizes))
    avg = np.zeros_like(vectors[0])
    for vector, size in zip(vectors, sizes):
        avg += vector * (size / total)
    return avg


def federated_train_eval_loop(
        model,
        client_train_datasets: list[tf.data.Dataset],
        client_eval_datasets: list[tf.data.Dataset],
        local_epochs: int,
        global_epochs: int) -> tuple[np.ndarray, list[float], list[list[list[float]]]]:
    """Simulate FedAvg over a single model instance.

    Each round: every client restores the current global weights, trains
    locally, and reports its weights; the server averages them (weighted by
    sample count) and evaluates. Weight transfer goes through the model's
    ``save``/``restore`` signatures - the exact mechanism used on-device.
    """
    num_clients = len(client_train_datasets)
    print(f"Starting federated training over {num_clients} clients...")

    client_sizes = [_count_batches(ds) for ds in client_train_datasets]
    global_weights = model.save()['parameters'].numpy()

    global_history: list[float] = []
    client_histories: list[list[list[float]]] = [[] for _ in range(num_clients)]

    for r in range(1, global_epochs + 1):
        print(f"\n--- Round {r}/{global_epochs} ---")

        local_vectors: list[np.ndarray] = []
        local_sizes: list[int] = []

        for cid, train_ds in enumerate(client_train_datasets):
            model.restore(tf.constant(global_weights))

            losses: list[float] = []
            for _ in range(local_epochs):
                epoch_loss, n_batches = 0.0, 0
                for signal, context, static in train_ds:
                    epoch_loss += float(model.train(signal, context, static)['loss'])
                    n_batches += 1
                losses.append(epoch_loss / max(n_batches, 1))

            local_vectors.append(model.save()['parameters'].numpy())
            local_sizes.append(client_sizes[cid])
            client_histories[cid].append(losses)
            print(f"    client {cid + 1}: local loss {losses[-1]:.6f}")

        global_weights = fed_avg(local_vectors, local_sizes)
        model.restore(tf.constant(global_weights))

        eval_loss = float(tf.reduce_mean([
            reconstruction_eval(model, ds) for ds in client_eval_datasets
        ]))
        global_history.append(eval_loss)
        print(f"    global eval loss: {eval_loss:.6f}")

    print("\nTraining complete.")
    model.restore(tf.constant(global_weights))
    return global_weights, global_history, client_histories

