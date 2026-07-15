import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.config import RESULTS_DIR
from ml.training import History

CALIBRATION_REPORT = 'distill_calibration.json'   # budgets, from distill_calibrate
EVAL_REPORT = 'distill_eval.json'                 # detector metrics, from distill_eval
RUN_MANIFEST = 'run.yaml'                         # run config + final metrics, from train.py


def load_budgets(model_name: str) -> dict[str, float]:
    """Read the global per-score budgets picked by distill_calibrate.py."""
    report_path = get_report_dir(model_name) / CALIBRATION_REPORT
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


def get_report_dir(model: str, subdir: str | None = None) -> Path:
    """Results directory for a model, rooted at RESULTS_DIR (created on demand).

    Holds everything evaluative — histories, detector metrics, convergence curves,
    figures — kept out of MODELS_DIR, which holds only the served .tflite artifacts.
    """
    report_dir = RESULTS_DIR / model
    if subdir is not None:
        report_dir = report_dir / subdir
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _plain(value):
    """Coerce numpy scalars and Paths into types yaml.safe_dump accepts."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_yaml(path: Path, fields: dict) -> None:
    path.write_text(yaml.safe_dump(_plain(fields), sort_keys=False, default_flow_style=False))


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def write_summary(path: Path, **fields) -> None:
    """Companion summary for a figure: `<name>.yaml` next to each `<name>.png`, so the
    result can be read and cited without opening the image. Keyword order is preserved;
    by convention start with `shows`, then the axes, the subjects/splits the numbers were
    measured on, the headline numbers, and what report section it backs."""
    write_yaml(path, fields)
    print(f"wrote summary to {path}")


def write_run(result_dir: Path, fields: dict) -> None:
    """Manifest of a train.py run: the configuration and final metrics behind the
    `training.csv` next to it. The figure scripts read this instead of re-running the
    loop, so it must carry everything they need to label and cross-check a curve."""
    path = result_dir / RUN_MANIFEST
    write_yaml(path, fields)
    print(f"wrote run manifest to {path}")


def read_run(model: str, loop: str) -> dict:
    """The manifest of a previous `train.py <model> --loop <loop>` run."""
    path = RESULTS_DIR / model / loop / RUN_MANIFEST
    if not path.exists():
        raise SystemExit(
            f"no {loop} run for '{model}' at {path}. Run "
            f"`uv run -m scripts.system.train {model} --loop {loop}` first — the figure "
            f"scripts plot a previous run's history, they do not train.")
    return read_yaml(path)


def read_history_csv(result_dir: Path) -> list[dict[str, float]]:
    """The `training.csv` a train.py run wrote, as one dict of floats per step."""
    path = result_dir / 'training.csv'
    if not path.exists():
        raise SystemExit(f"no training history at {path}")
    with path.open(newline='') as f:
        return [{k: float(v) for k, v in row.items() if v != ''}
                for row in csv.DictReader(f)]
