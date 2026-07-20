import csv
from pathlib import Path
import numpy as np
import yaml

from common.config import RESULTS_DIR

RUN_MANIFEST = 'run.yaml'   # run config + final metrics, from train.py


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


def loop_dir(loop: str, tag: str | None = None) -> str:
    return f'{loop}_{tag}' if tag else loop


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
    """Writes a run manifest, or a figure's companion summary — `<name>.yaml` next to
    each `<name>.png`, so the result can be read and cited without opening the image.
    Key order is preserved; by convention a summary starts with `shows`, then the axes,
    the subjects/splits the numbers were measured on, the headline numbers"""
    path.write_text(yaml.safe_dump(_plain(fields), sort_keys=False, default_flow_style=False))
    print(f"wrote {path}")


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def read_subject_split(model: str, loops: tuple[str, ...],
                       tag: str | None = None) -> tuple[list[str], list[str]]:
    """The (train_ids, eval_ids) a previous train.py run recorded — the exact held-out
    subjects, not merely a count, since the selection may be arbitrary."""
    for loop in loops:
        path = RESULTS_DIR / model / loop_dir(loop, tag) / RUN_MANIFEST
        if path.exists():
            run = read_yaml(path)
            eval_ids = run['eval_subjects']
            if not eval_ids:
                raise SystemExit(
                    f"'{model}' {loop_dir(loop, tag)} run held out no subjects (all-users "
                    f"teacher); it has no held-out set to score. Train a split run first.")
            return run['train_subjects'], eval_ids
    tag_flag = f' --tag {tag}' if tag else ''
    raise SystemExit(
        f"no run manifest for '{model}' under {[loop_dir(l, tag) for l in loops]}; run "
        f"`uv run -m scripts.system.train {model}{tag_flag}` first so the held-out split "
        f"is recorded.")


def read_run(model: str, loop: str, tag: str | None = None) -> dict:
    """The manifest of a previous `train.py <model> --loop <loop>` run. The figure
    scripts read this instead of re-running the loop, so it must carry everything they
    need to label and cross-check a curve."""
    path = RESULTS_DIR / model / loop_dir(loop, tag) / RUN_MANIFEST
    if not path.exists():
        tag_flag = f' --tag {tag}' if tag else ''
        raise SystemExit(
            f"no {loop_dir(loop, tag)} run for '{model}' at {path}. Run "
            f"`uv run -m scripts.system.train {model} --loop {loop}{tag_flag}` first — the "
            f"figure scripts plot a previous run's history, they do not train.")
    return read_yaml(path)


def write_metrics_csv(rows: list[dict], result_dir: Path, name: str):
    fields = list(rows[0]) if rows else []
    path = result_dir / name
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def write_history_csv(history, result_dir: Path):
    metric_keys = sorted({k for _, _, metrics in history for k in metrics})
    path = result_dir / 'training.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', *metric_keys])
        for step, loss, metrics in history:
            writer.writerow([step, loss, *(metrics.get(k, '') for k in metric_keys)])
    print(f"saved training history to {path}")


def read_history_csv(result_dir: Path) -> list[dict[str, float]]:
    """The `training.csv` a train.py run wrote, as one dict of floats per step."""
    path = result_dir / 'training.csv'
    if not path.exists():
        raise SystemExit(f"no training history at {path}")
    with path.open(newline='') as f:
        return [{k: float(v) for k, v in row.items() if v != ''}
                for row in csv.DictReader(f)]
