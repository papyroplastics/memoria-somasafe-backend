from pathlib import Path
import numpy as np
import tensorflow as tf

from common.config import MODELS_DIR
from ml.saving import load_trainable_weights
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer

def window_errors(model, signal: np.ndarray, cond: np.ndarray,
                  window: int, n_windows: int) -> np.ndarray:
    """Per-window reconstruction error for non-overlapping ``window``-sample frames.

    ``n_windows`` is an upper bound (e.g. the label count); the actual count is
    shrunk to whole windows that fit the signal and to a multiple of the batch
    size, dropping the trailing remainder. Returns errors of that (possibly
    smaller) length, so callers must align their labels to it."""
    bs = model.batch_size
    signal = signal.astype(np.float32)
    cond = cond.astype(np.float32)
    n_windows = min(n_windows, len(signal) // window, len(cond))
    n_windows -= n_windows % bs
    errors = np.empty(n_windows, dtype=np.float32)
    for start in range(0, n_windows, bs):
        wins = np.stack([signal[(start + i) * window:(start + i + 1) * window] for i in range(bs)])
        conds = cond[start:start + bs]
        out = model.eval(wins, conds)
        errors[start:start + bs] = out['error'].numpy()
    return errors


def load_autoencoder(model_name: str, batch_size: int | None = None) -> AutoencoderTrainer:
    """Build an autoencoder trainer and restore its trained
    weights from results/<model>/trainable.tflite."""
    trainer = MODELS[model_name].build_trainer(batch_size)
    if not isinstance(trainer, AutoencoderTrainer):
        raise SystemExit(
            f"'{model_name}' is not an autoencoder; testing needs one (lstm-ae, gru-ae, cnn-ae).")

    weights_path = MODELS_DIR / model_name / 'trainable.tflite'
    if not weights_path.exists():
        raise SystemExit(f"trained model not found at {weights_path}. Train '{model_name}' first.")
    trainer.model.restore(tf.constant(load_trainable_weights(weights_path)))
    return trainer


