from pathlib import Path
import numpy as np
import tensorflow as tf

from common.config import RESULTS_DIR
from ml.saving import load_trainable_weights
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer

def window_errors(model, signal: np.ndarray, cond: np.ndarray,
                  window: int, n_windows: int) -> np.ndarray:
    """Reconstruction error for the first ``n_windows`` non-overlapping windows of a
    normalized ``[BVP, ACC]`` signal, each scored with its conditioning vector.
    Built with a batch-size-1 model so each window scores independently."""
    errors = np.empty(n_windows, dtype=np.float32)
    for w in range(n_windows):
        s = w * window
        win = signal[s:s + window]
        out = model.eval(win[None].astype(np.float32), cond[w][None].astype(np.float32))
        errors[w] = float(out['error'][0])
    return errors


def load_autoencoder(model_name: str) -> AutoencoderTrainer:
    """Build a batch-size-1 autoencoder trainer and restore its trained weights from
    results/<model>/trainable.tflite."""
    trainer = MODELS[model_name].build_trainer(batch_size=1)
    if not isinstance(trainer, AutoencoderTrainer):
        raise SystemExit(
            f"'{model_name}' is not an autoencoder; testing needs one (lstm-ae, gru-ae, cnn-ae).")

    weights_path = RESULTS_DIR / model_name / 'trainable.tflite'
    if not weights_path.exists():
        raise SystemExit(f"trained model not found at {weights_path}. Train '{model_name}' first.")
    trainer.model.restore(tf.constant(load_trainable_weights(weights_path)))
    return trainer


