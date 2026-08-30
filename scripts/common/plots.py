from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.training import History


def line_plot(path: Path, x: list, series: dict[str, list], xlabel: str, ylabel: str,
              title: str | None = None, marker: str = 'o-',
              vline: tuple[float, str] | None = None, logx: bool = False,
              diagonal: bool = False) -> None:
    """Plots one or more series against a shared x axis."""
    fig, ax = plt.subplots()
    markers = [marker, 's-', '^-', 'd-']
    for i, (name, values) in enumerate(series.items()):
        ax.plot(x[:len(values)], values, markers[i % len(markers)],
                label=name if len(series) > 1 else None)
    if diagonal:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='random classifier')
    if vline is not None:
        ax.axvline(vline[0], color='k', linestyle='--', linewidth=1, label=vline[1])
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1 or vline is not None or diagonal:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


def roc_grid(path: Path, order: list[str], curves: dict[str, tuple[list, list]],
             highlight: set[str], xlabel: str, ylabel: str, title: str,
             ncols: int = 5) -> None:
    """Draws one small ROC panel per subject on a shared grid."""
    nrows = -(-len(order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows),
                             sharex=True, sharey=True)
    for ax, sid in zip(axes.flat, order):
        fpr, recall = curves[sid]
        held = sid in highlight
        ax.plot(fpr, recall, color='C3' if held else 'C0')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{sid} (held-out)' if held else sid, fontsize=9)
    for ax in axes.flat[len(order):]:
        ax.axis('off')
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


def bar_plot(path: Path, x: list, values: list[float] | dict[str, list[float]],
             xlabel: str, ylabel: str, title: str, mean_line: float | None = None,
             hlines: list[tuple[float, str]] | None = None) -> None:
    """Draws one bar per x, or several grouped bars per x when ``values`` is a dict of named series."""
    fig, ax = plt.subplots()
    if isinstance(values, dict):
        width = 0.8 / len(values)
        offsets = np.arange(len(x)) - 0.4 + width / 2
        for i, (name, series) in enumerate(values.items()):
            ax.bar(offsets + i * width, series, width, label=name)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
    else:
        ax.bar(x, values)
    if mean_line is not None:
        ax.axhline(mean_line, color='k', linestyle='--', label=f'mean {mean_line:.4f}')
    for i, (y, label) in enumerate(hlines or []):
        ax.axhline(y, color='k', linestyle=(':' if i else '--'), linewidth=1, label=label)
    if mean_line is not None or hlines or isinstance(values, dict):
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved plot to {path}")


def plot_history(history: History, primary_metric: str, result_dir: Path) -> None:
    """Plots training loss and the held-out metric against the step, on twin y axes."""
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
