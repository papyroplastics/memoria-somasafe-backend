"""
Plot one random 8-second window from a random subject for the
clean signal and every anomaly kind, then a second figure with
those same windows reconstructed by a trained autoencoder.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.config import DATASETS_DIR, RESULTS_DIR
from ml.model_list import MODELS
from ml.data import (
    CLEAN_SUBDIR, ANOMALOUS_SUBDIR, ANOMALY_KINDS, BVP_RATE,
    conditional_windows, get_sorted_paths,
)
from .common.post_train import get_report_dir
from .common.autoencoders import load_autoencoder

KINDS = ('clean', *ANOMALY_KINDS)


def window_views(data_dir, sid, window, index):
    """Normalized [BVP, ACC] window + its conditioning vector for the clean signal
    and each anomaly kind, all sliced at the same window ``index``. Only BVP carries
    the anomaly; ACC is always the subject's clean signal."""
    subjects_dir = data_dir / CLEAN_SUBDIR
    anomalous_dir = data_dir / ANOMALOUS_SUBDIR

    views = {}
    for kind in KINDS:
        src = None if kind == 'clean' else anomalous_dir / kind
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=src)
        views[kind] = (signal[index * window:(index + 1) * window], cond[index])
    return views


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to use')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed for the subject/window pick')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    trainer = load_autoencoder(args.model, batch_size=1)
    window = trainer.window_size

    subjects_dir = DATASETS_DIR / CLEAN_SUBDIR
    subject_dirs = get_sorted_paths(subjects_dir)
    if not subject_dirs:
        raise SystemExit(f"{subjects_dir} is empty. Run get_dataset.py first.")

    sid = subject_dirs[rng.integers(len(subject_dirs))].name
    _, cond = conditional_windows(subjects_dir, sid, window)
    n_windows = len(cond)
    if n_windows == 0:
        raise SystemExit(f"{sid} has no full {window}-sample window.")
    index = int(rng.integers(n_windows))
    print(f"subject={sid} window={index}/{n_windows}")

    views = window_views(DATASETS_DIR, sid, window, index)
    t = np.arange(window) / BVP_RATE

    fig_in, axs_in = plt.subplots(len(KINDS), 1, sharex=True, figsize=(8, 2 * len(KINDS)))
    fig_rec, axs_rec = plt.subplots(len(KINDS), 1, sharex=True, figsize=(8, 2 * len(KINDS)))
    fig_in.suptitle(f'{sid} window {index} — normalized BVP')
    fig_rec.suptitle(f'{sid} window {index} — {args.model} reconstruction')

    for ax_in, ax_rec, kind in zip(axs_in, axs_rec, KINDS):
        sig, cond = views[kind]
        bvp = sig[:, 0]
        recon = trainer.model.eval(
            sig[None].astype(np.float32),
            cond[None].astype(np.float32),
        )['reconstruction'][0, :, 0].numpy()

        ax_in.plot(t, bvp)
        ax_in.set_ylabel(kind)

        ax_rec.plot(t, bvp, alpha=0.4, label='input')
        ax_rec.plot(t, recon, label='reconstruction')
        ax_rec.set_ylabel(kind)

    axs_in[-1].set_xlabel('seconds')
    axs_rec[-1].set_xlabel('seconds')
    axs_rec[0].legend(loc='upper right')

    report_dir = get_report_dir(RESULTS_DIR / args.model)
    in_path = report_dir / 'signals.png'
    rec_path = report_dir / 'signals_reconstructed.png'
    fig_in.savefig(in_path)
    fig_rec.savefig(rec_path)
    print(f"saved input windows to {in_path}")
    print(f"saved reconstructions to {rec_path}")
