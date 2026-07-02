from pathlib import Path
import matplotlib.pyplot as plt

from ml.training import History

AE_TEST_REPORT = 'autoencoder_test.json'


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


