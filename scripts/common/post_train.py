from pathlib import Path
import matplotlib.pyplot as plt

from ml.training import History
from ml.models.common import Trainer
from ml.saving import save_tainable_model, save_optimized_model

# Autoencoder evaluation report (threshold + metrics), written by test_autoencoder.py
# into results/<model>/reports/ and read back by distill_labels.py.
AE_TEST_REPORT = 'autoencoder_test.json'

def save_artifacts(trainer: Trainer, result_dir: Path, eval_dataset, postfix: str = ''):
    saved_model, sm_path = save_tainable_model(result_dir, trainer.model, postfix)
    print(f"Saved trainable model to {sm_path}")
    try:
        rep_dataset = trainer.representative_dataset(eval_dataset)
        save_optimized_model(result_dir, trainer.model, rep_dataset, postfix)
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

def get_report_dir(result_dir: Path) -> Path:
    report_dir = result_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


