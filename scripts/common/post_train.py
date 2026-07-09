import csv
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


def write_history_csv(history: History, result_dir: Path):
    metric_keys = sorted({k for _, _, metrics in history for k in metrics})
    path = result_dir / 'training.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', *metric_keys])
        for step, loss, metrics in history:
            writer.writerow([step, loss, *(metrics.get(k, '') for k in metric_keys)])
    print(f"saved training history to {path}")


def write_metrics_csv(rows: list[dict], result_dir: Path, name: str):
    fields = list(rows[0]) if rows else []
    path = result_dir / name
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def plot_metric(rows: list[dict], x_key: str, y_key: str, result_dir: Path, name: str):
    fig, ax = plt.subplots()
    ax.plot([r[x_key] for r in rows], [r[y_key] for r in rows], 'g-', label=y_key)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    path = result_dir / name
    fig.savefig(path)
    print(f"saved plot to {path}")


def get_report_dir(result_dir: Path, subdir: str | None = None) -> Path:
    report_dir = result_dir / 'reports'
    if subdir is not None:
        report_dir = report_dir / subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir
