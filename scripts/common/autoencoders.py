from pathlib import Path
import numpy as np
import tensorflow as tf

from common.config import RESULTS_DIR
from ml.saving import load_trainable_weights
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer

def window_errors(model, signal: np.ndarray, cond: np.ndarray,
                  window: int, n_windows: int) -> np.ndarray:
    bs = model.batch_size
    signal = signal.astype(np.float32)
    cond = cond.astype(np.float32)
    errors = np.empty(n_windows, dtype=np.float32)
    for start in range(0, n_windows, bs):
        count = min(bs, n_windows - start)
        wins = np.stack([signal[(start + i) * window:(start + i + 1) * window]
                         for i in range(count)])
        conds = cond[start:start + count]
        if count < bs:   # pad the final batch up to the static batch size
            wins = np.concatenate([wins, np.zeros((bs - count, *wins.shape[1:]), np.float32)])
            conds = np.concatenate([conds, np.zeros((bs - count, *conds.shape[1:]), np.float32)])
        out = model.eval(wins, conds)
        errors[start:start + count] = out['error'].numpy()[:count]
    return errors


def load_autoencoder(model_name: str) -> AutoencoderTrainer:
    """Build an autoencoder trainer and restore its trained
    weights from results/<model>/trainable.tflite."""
    trainer = MODELS[model_name].build_trainer()
    if not isinstance(trainer, AutoencoderTrainer):
        raise SystemExit(
            f"'{model_name}' is not an autoencoder; testing needs one (lstm-ae, gru-ae, cnn-ae).")

    weights_path = RESULTS_DIR / model_name / 'trainable.tflite'
    if not weights_path.exists():
        raise SystemExit(f"trained model not found at {weights_path}. Train '{model_name}' first.")
    trainer.model.restore(tf.constant(load_trainable_weights(weights_path)))
    return trainer


