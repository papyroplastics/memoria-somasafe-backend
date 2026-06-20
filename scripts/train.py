import argparse
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

from ml.models.common import Trainer
from ml.training import normal_loop, federated_loop, History
from ml.saving import save_tainable_model, save_optimized_model
from ml.models import (
    lstm_autoencoder,
    cond_lstm_autoencoder,
    gru_autoencoder,
    cnn_autoencoder,
    feature_mlp
)

SEED = 1234

# Each model module exposes get_trainer(data_root, seed) -> Trainer.
TRAINERS = {
    'feature-mlp': feature_mlp.get_trainer,
    'lstm-ae': lstm_autoencoder.get_trainer,
    'cond-lstm-ae': cond_lstm_autoencoder.get_trainer,
    'gru-ae': gru_autoencoder.get_trainer,
    'cnn-ae': cnn_autoencoder.get_trainer,
}


def run_loop(trainer: Trainer, data_dir: Path, loop: str, 
             epochs: int, local_epochs: int) -> tuple[History, tf.data.Dataset]:
    subject_train, subject_eval = trainer.subject_datasets(data_dir, SEED)
    eval_dataset = trainer.combine(subject_eval)

    if loop == 'normal':
        train_dataset = trainer.combine(subject_train)
        history = normal_loop(trainer, train_dataset, eval_dataset, epochs)
    else:
        history = federated_loop(trainer, subject_train, eval_dataset,
                                 local_epochs, epochs)
    return history, eval_dataset


def save_artifacts(trainer: Trainer, result_dir: Path, eval_dataset):
    saved_model, sm_path = save_tainable_model(result_dir, trainer.model)
    print(f"Saved trainable model to {sm_path}")
    try:
        rep_dataset = trainer.representative_dataset(eval_dataset)
        save_optimized_model(result_dir, trainer.model, rep_dataset)
    except Exception as e:
        print(f"Skipped int8 export (conversion failed): {e}")


def plot_history(history: History, primary_metric: str, result_dir: Path):
    steps = [h[0] for h in history]
    losses = [h[1] for h in history]
    metric = [h[2][primary_metric] for h in history]

    fig, ax = plt.subplots()
    ax.plot(steps, losses, 'b-', label='train loss')
    ax.set_xlabel('step')
    ax.set_ylabel('loss', color='b')
    ax2 = ax.twinx()
    ax2.plot(steps, metric, 'g-', label=primary_metric)
    ax2.set_ylabel(primary_metric, color='g')
    fig.savefig(result_dir / 'training.png')
    print(f"saved training plot to {result_dir / 'training.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a SomaSafe model with a chosen training loop and export '
                    'SavedModel + TFLite artifacts into results/<model>.')
    parser.add_argument('model', choices=sorted(TRAINERS), help='Model to train')
    parser.add_argument('--loop', choices=['normal', 'federated'], default='normal',
                        help='Training loop (default: normal)')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs for the normal loop')
    parser.add_argument('--local-epochs', type=int, default=5, help='Local epochs per round (federated)')
    args = parser.parse_args()

    data_dir = Path('datasets')
    result_dir = Path('results') / args.model
    result_dir.mkdir(parents=True, exist_ok=True)

    trainer = TRAINERS[args.model](data_dir, SEED)
    history, eval_dataset = run_loop(trainer, data_dir, args.loop, args.epochs, args.local_epochs)

    save_artifacts(trainer, result_dir, eval_dataset)
    plot_history(history, trainer.primary_metric, result_dir)
    trainer.report(result_dir, eval_dataset)
