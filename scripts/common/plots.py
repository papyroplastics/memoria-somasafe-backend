from pathlib import Path

import matplotlib.pyplot as plt

from ml.training import History


def line_plot(path: Path, x: list, series: dict[str, list], xlabel: str, ylabel: str,
              title: str | None = None, marker: str = 'o-',
              vline: tuple[float, str] | None = None, logx: bool = False) -> None:
    """One or more series against a shared x axis. A single series is drawn unlabeled
    (no legend); several get a legend keyed by name. ``vline`` marks an x position with
    a labelled dashed rule (e.g. the operating point a sweep selected)."""
    fig, ax = plt.subplots()
    markers = [marker, 's-', '^-', 'd-']
    for i, (name, values) in enumerate(series.items()):
        ax.plot(x[:len(values)], values, markers[i % len(markers)],
                label=name if len(series) > 1 else None)
    if vline is not None:
        ax.axvline(vline[0], color='k', linestyle='--', linewidth=1, label=vline[1])
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1 or vline is not None:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


def bar_plot(path: Path, x: list, values: list[float], xlabel: str, ylabel: str,
             title: str, mean_line: float | None = None) -> None:
    fig, ax = plt.subplots()
    ax.bar(x, values)
    if mean_line is not None:
        ax.axhline(mean_line, color='k', linestyle='--', label=f'mean {mean_line:.4f}')
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


def plot_history(history: History, primary_metric: str, result_dir: Path) -> None:
    """Training loss and the held-out metric against the step, on twin y axes — they
    have unrelated scales."""
    steps = [h[0] for h in history]

    fig, ax = plt.subplots()
    ax.plot(steps, [h[1] for h in history], 'b-', label='train loss')
    ax.set_xlabel('step')
    ax.set_ylabel('loss', color='b')
    ax2 = ax.twinx()
    ax2.plot(steps, [h[2][primary_metric] for h in history], 'g-', label=primary_metric)
    ax2.set_ylabel(primary_metric, color='g')
    path = result_dir / 'training.png'
    fig.savefig(path)
    plt.close(fig)
    print(f"saved training plot to {path}")
