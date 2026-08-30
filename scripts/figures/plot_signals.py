"""
Plot one 8-second window from a subject for the clean signal and every anomaly kind, then a
second figure with those same windows reconstructed by a trained autoencoder.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ..common.scoring import eval_padded
from ml.dataset_list import DATASETS
from ml.model_list import MODELS
from ml.preprocessing import ANOMALY_KINDS, BVP_RATE
from ml.saving import load_weights, weights_path
from ml.sources.common import DataSource
from ml.sources.dalia import CLEAN
from ..common.reports import get_report_dir, write_yaml

KINDS = (CLEAN, *ANOMALY_KINDS)


def window_views(source: DataSource, sid: str, window: int, index: int):
    """The raw BVP window for the clean signal and each anomaly kind, all taken at the
    same window ``index`` of the windows the source keeps."""
    return {kind: source.signal_windows(sid, kind, window, window)[index]
            for kind in KINDS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to use')
    parser.add_argument('--subject', type=int, default=None, help='Subject to use')
    parser.add_argument('--window', type=int, default=None,
                        help='Window index to use, counted over the windows the dataset keeps')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed for the subject/window pick')
    parser.add_argument('--tag', default=None,
                        help='Tag of the train.py run to use')
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='ppg-dalia',
                        help='Dataset the window is drawn from; ppg-dalia-low draws only '
                             'from low-activity windows')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    source = DATASETS[args.dataset].build(DATASETS_DIR)

    model = MODELS[args.model].build_model(DATASETS_DIR)
    if not hasattr(model, 'seq_len'):
        raise SystemExit(f"{args.model} does not reconstruct a waveform, so there is "
                         f"nothing to overlay on one — this plot is for the LSTM/GRU/CNN "
                         f"autoencoders")
    weights = weights_path(MODELS_DIR / args.model, args.tag)
    model.restore(load_weights(weights))

    window_len = model.seq_len

    subject_ids = source.subject_ids()
    sid = f"S{args.subject}" if args.subject else str(rng.choice(subject_ids))
    if sid not in subject_ids:
        raise SystemExit(f"subject {sid} not found among {subject_ids}")

    n_windows = len(source.signal_windows(sid, CLEAN, window_len, window_len))
    if not n_windows:
        raise SystemExit(f"{sid} has no windows in {DATASETS[args.dataset].name}")

    window_idx = args.window if args.window is not None else int(rng.integers(n_windows))
    print(f"dataset={args.dataset} subject={sid} window={window_idx}/{n_windows}")

    views = window_views(source, sid, window_len, window_idx)
    t = np.arange(window_len) / BVP_RATE

    signals = np.stack([views[k] for k in KINDS]).astype(np.float32)
    recons = eval_padded(model, signals)['reconstruction'][:, :, 0]

    fig_in, axs_in = plt.subplots(len(KINDS), 1, sharex=True, figsize=(8, 2 * len(KINDS)))
    fig_rec, axs_rec = plt.subplots(len(KINDS), 1, sharex=True, figsize=(8, 2 * len(KINDS)))
    fig_in.suptitle(f'{sid} window {window_idx} — BVP, z-scored on {sid}')
    fig_rec.suptitle(f'{sid} window {window_idx} — {args.model} reconstruction')

    for i, (ax_in, ax_rec, kind) in enumerate(zip(axs_in, axs_rec, KINDS)):
        # Both the window and the reconstruction are in the subject's z-scored space:
        # the source normalizes on the way out and the model neither un- nor re-scales.
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

    sample = {'dataset': args.dataset, 'subject': sid, 'window': window_idx,
              'of_windows': n_windows, 'seed': args.seed}
    axes = {'x_axis': {'name': 'seconds', 'range': [0, 8], 'sample_rate_hz': BVP_RATE},
            'y_axis': {'name': 'BVP amplitude',
                       'units': "z-scores of the subject's own clean signal"}}

    write_yaml(report_dir / 'signals.yaml', {
        'shows': f"BVP signal windows for subject {sid}: one 8 s window per row, the "
                 f"same window under the clean signal and each synthetic anomaly kind.",
        'rows': {'order': 'top to bottom', 'kinds': list(KINDS)},
        **axes,
        'sample': sample,
        'note': "anomalies are injected into BVP only",
    })
    write_yaml(report_dir / 'signals_reconstructed.yaml', {
        'shows': f"The same {len(KINDS)} windows with the {args.model} autoencoder's "
                 f"reconstruction overlaid on the input, both in the subject's own "
                 f"z-scored units: "
                 f"the autoencoder tracks clean rhythm and departs on integrity/rhythm "
                 f"anomalies.",
        'rows': {'order': 'top to bottom', 'kinds': list(KINDS)},
        **axes,
        'sample': sample,
    })
