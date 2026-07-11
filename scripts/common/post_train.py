import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

from common.config import MODELS_DIR
from ml.training import History

CALIBRATION_REPORT = 'distill_calibration.json'   # budgets, from distill_calibrate
EVAL_REPORT = 'distill_eval.json'                 # detector metrics, from distill_eval


def load_budgets(model_name: str) -> dict[str, float]:
    """Read the global per-score budgets picked by distill_calibrate.py."""
    report_path = get_report_dir(MODELS_DIR / model_name) / CALIBRATION_REPORT
    if not report_path.exists():
        raise SystemExit(
            f"no calibration report at {report_path}. Run distill_calibrate '{model_name}' "
            f"first to pick the budgets.")
    return {k: float(v) for k, v in json.loads(report_path.read_text())['budgets'].items()}


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
