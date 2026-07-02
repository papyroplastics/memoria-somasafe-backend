import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from ml.training import History
from ml.models.common import Trainer
from ml.saving import save_tainable_model, save_optimized_model
from ml.data import (
    CLEAN_SUBDIR, norm_stats, load_context_norm_params, load_static_norm_params,
)

AE_TEST_REPORT = 'autoencoder_test.json'
NORM_PARAMS_JSON = 'norm.json'

def save_artifacts(trainer: Trainer, result_dir: Path,
                   eval_dataset: tf.data.Dataset | None, postfix: str = ''):
    saved_model, sm_path = save_tainable_model(result_dir, trainer.model, postfix)
    print(f"Saved trainable model to {sm_path}")
    try:
        rep_dataset = trainer.representative_dataset(eval_dataset)
        save_optimized_model(result_dir, trainer.model, rep_dataset, postfix)
    except Exception as e:
        print(f"Skipped int8 export (conversion failed): {e}")


def stage_norm_params(result_dir: Path, data_dir: Path):
    """Copy the dataset-global normalization params the on-device trainer needs
    (signal / context / static mean-std) into the model dir as norm.json, so the
    gateway can serve them per model over /model/norm. The std values already carry
    the load-time EPS, so the device applies (x - mean) / std verbatim."""
    subjects_dir = data_dir / CLEAN_SUBDIR
    sig_mean, sig_std = norm_stats(subjects_dir)
    ctx_mean, ctx_std = load_context_norm_params(subjects_dir)
    stat_mean, stat_std = load_static_norm_params(subjects_dir)
    payload = {
        'signal_mean': sig_mean.tolist(),  'signal_std': sig_std.tolist(),
        'context_mean': ctx_mean.tolist(), 'context_std': ctx_std.tolist(),
        'static_mean': stat_mean.tolist(), 'static_std': stat_std.tolist(),
    }
    (result_dir / NORM_PARAMS_JSON).write_text(json.dumps(payload))
    print(f"staged norm params to {result_dir / NORM_PARAMS_JSON}")


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

def get_report_dir(result_dir: Path) -> Path:
    report_dir = result_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


