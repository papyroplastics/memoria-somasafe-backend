"""
Plot one 8-second window from a subject for the clean signal and every anomaly kind, then a
second figure with those same windows reconstructed by a trained autoencoder.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ..common.scoring import eval_padded, variant_sources
from ml.dataset_list import BASES
from ml.model_list import MODELS
from ml.sources.common import DataSource
from ml.sources.dalia import ANOMALY_KINDS, BVP_RATE, CLEAN
from ml.saving import load_weights, weights_path
from ..common.reports import get_report_dir, write_yaml

KINDS = (CLEAN, *ANOMALY_KINDS)


def window_views(sources: dict[str, DataSource], sid: str, index: int):
    """The raw BVP window for the clean signal and each anomaly kind, all taken at the
    same window ``index`` of the windows each source keeps."""
    return {kind: sources[kind].datapoints(sid)[index] for kind in KINDS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to use')
    parser.add_argument('--subject', type=int, default=None, help='Subject to use')
    parser.add_argument('--window', type=int, default=None,
                        help='Window index to use, counted over the windows the dataset keeps')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed for the subject/window pick')
    parser.add_argument('--tag', default=None,
                        help='Tag of the train.py run to use')
    parser.add_argument('--dataset', choices=sorted(BASES), default='ppg-dalia',
                        help='Dataset the window is drawn from; ppg-dalia-low draws only '
                             'from low-activity windows')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    model = MODELS[args.model].model_cls()
    if not hasattr(model, 'seq_len'):
        raise SystemExit(f"{args.model} does not reconstruct a waveform, so there is "
                         f"nothing to overlay on one — this plot is for the LSTM/GRU/CNN "
                         f"autoencoders")
    weights = weights_path(MODELS_DIR / args.model, args.tag)
    model.restore(load_weights(weights))

    window_len = model.seq_len
    sources = {k: s.with_grid(window=window_len, shift=window_len) for k, s in
              variant_sources(model, args.dataset, DATASETS_DIR, variants=KINDS).items()}

    subject_ids = sources[CLEAN].subject_ids()
    sid = f"S{args.subject}" if args.subject else str(rng.choice(subject_ids))
    if sid not in subject_ids:
        raise SystemExit(f"subject {sid} not found among {subject_ids}")

    n_windows = len(sources[CLEAN].datapoints(sid))
    if not n_windows:
        raise SystemExit(f"{sid} has no windows in {BASES[args.dataset].name}")

    window_idx = args.window if args.window is not None else int(rng.integers(n_windows))
    print(f"dataset={args.dataset} subject={sid} window={window_idx}/{n_windows}")

    views = window_views(sources, sid, window_idx)
    t = np.arange(window_len) / BVP_RATE

    signals = np.stack([views[k] for k in KINDS]).astype(np.float32)
    recons = eval_padded(model, signals)['reconstruction'][:, :, 0]

    fig_in, axs_in = plt.subplots(len(KINDS), 1, sharex=True, figsize=(8, 2 * len(KINDS)))
    fig_rec, axs_rec = plt.subplots(len(KINDS), 1, sharex=True, figsize=(8, 2 * len(KINDS)))
    fig_in.suptitle(f'{sid} window {window_idx} — BVP, z-scored on {sid}')
    fig_rec.suptitle(f'{sid} window {window_idx} — {args.model} reconstruction')

    for i, (ax_in, ax_rec, kind) in enumerate(zip(axs_in, axs_rec, KINDS)):
        bvp = views[kind][:, 0]
        recon = recons[i]

        ax_in.plot(t, bvp)
        ax_in.set_ylabel(kind)

        ax_rec.plot(t, bvp, alpha=0.4, label='input')
        ax_rec.plot(t, recon, label='reconstruction')
        ax_rec.set_ylabel(kind)

    axs_in[-1].set_xlabel('seconds')
    axs_rec[-1].set_xlabel('seconds')
    axs_rec[0].legend(loc='upper right')

    report_dir = get_report_dir(args.model)
    in_path = report_dir / 'signals.png'
    rec_path = report_dir / 'signals_reconstructed.png'
    fig_in.savefig(in_path)
    fig_rec.savefig(rec_path)
    print(f"saved input windows to {in_path}")
    print(f"saved reconstructions to {rec_path}")

    sample = {'dataset': args.dataset, 'model': args.model, 'subject': sid,
              'window': window_idx, 'of_windows': n_windows, 'seed': args.seed,
              'kinds': list(KINDS)}

    write_yaml(report_dir / 'signals.yaml', sample)
    write_yaml(report_dir / 'signals_reconstructed.yaml', sample)
