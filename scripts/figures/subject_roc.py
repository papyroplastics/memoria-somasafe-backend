"""
Per-subject detectability of the autoencoder detector, scored on one set of weights.

Unlike ``calibrate_fpr`` (which tells the split-teacher calibration story on 2 held-out
subjects), this scores **every** subject on the given model and lays their ROC curves on a
shared grid, so per-user detectability is comparable at a glance — the answer to "does this
model just catch some users better than others". Point ``--weights`` at an all-users teacher
(a renamed ``trainable_all.tflite``) to put every subject on equal footing; note that then
every subject was trained on, so this is the *population spread*, not a generalization number.

Each subject's threshold is the ``1 - f`` quantile of its own clean scores (as everywhere in
the detector), swept over ``f``. Two figures + a table land under ``results/<model>/subject_roc/``:

    roc_by_subject.png   one ROC panel per subject (recall vs. empirical clean FPR)
    roc_aggregate.png    mean +/- std recall across subjects vs. expected FPR
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.config import DATASETS_DIR, MODELS_DIR
from ml.model_list import MODELS
from ml.preprocessing import CLEAN_SUBDIR, MIXED_SUBDIR
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights

from ..common.plots import roc_grid
from ..common.reports import get_report_dir, write_yaml
from ..common.scoring import DETECTOR, score_dir_by_subject, load_mixed_truth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to score')
    parser.add_argument('--weights', type=Path, default=None,
                        help='Trainable .tflite to score (default: the canonical '
                             'trainable.tflite). Point at an all-users teacher to compare '
                             'every subject on equal footing.')
    parser.add_argument('--highlight', default='',
                        help="Comma-separated subject ids to draw in red (e.g. a split's "
                             "held-out pair), so you can see where they sit in the pack.")
    parser.add_argument('--step', type=float, default=0.02,
                        help='Spacing of the FPR sweep (default: 0.02)')
    args = parser.parse_args()

    weights = args.weights or (MODELS_DIR / args.model / 'trainable.tflite')
    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    trainer.model.restore(load_trainable_weights(weights))
    assert isinstance(trainer, AutoencoderTrainer)

    clean = score_dir_by_subject(trainer, DATASETS_DIR / CLEAN_SUBDIR)
    mixed = score_dir_by_subject(trainer, DATASETS_DIR / MIXED_SUBDIR)
    truth = load_mixed_truth(DATASETS_DIR, mixed)
    order = [sid for sid in clean if sid in mixed and sid in truth]
    highlight = {f'S{int(i)}' for i in args.highlight.split(',') if i.strip()}

    fine = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 4)
    curves, per_subject, recalls = {}, {}, []
    for sid in order:
        c, m, t = clean[sid][DETECTOR], mixed[sid][DETECTOR], truth[sid]
        thr = np.quantile(c, 1.0 - fine)                       # one threshold per FPR level
        fpr = (c[:, None] > thr).mean(axis=0)
        anom = m[t == 1]
        recall = (anom[:, None] > thr).mean(axis=0) if len(anom) else np.full_like(fpr, np.nan)
        curves[sid] = (fpr.tolist(), recall.tolist())
        recalls.append(recall)
        per_subject[sid] = {'auc': float(np.trapezoid(recall, fpr)),
                            'anomalous_windows': int((t == 1).sum())}

    report_dir = get_report_dir(args.model, 'subject_roc')
    roc_grid(report_dir / 'roc_by_subject.png', order, curves, highlight,
             'empirical clean FPR', 'recall',
             f'{args.model} — per-subject ROC ({weights.name})')

    stack = np.vstack(recalls)
    mean, std = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
    fig, ax = plt.subplots()
    ax.plot(fine, mean, 'C0-', label='mean recall')
    ax.fill_between(fine, mean - std, mean + std, alpha=0.2, color='C0', label='±1 std')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='random classifier')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('expected FPR')
    ax.set_ylabel('recall')
    ax.set_title(f'{args.model} — recall vs. expected FPR, mean ± std over {len(order)} subjects')
    ax.legend()
    fig.savefig(report_dir / 'roc_aggregate.png')
    plt.close(fig)
    print(f"saved plot to {report_dir / 'roc_aggregate.png'}")

    aucs = [s['auc'] for s in per_subject.values()]
    write_yaml(report_dir / 'subject_roc.yaml', {
        'shows': "Per-subject detectability of the reconstruction-error detector on one set "
                 "of weights: each subject's ROC (recall vs. its own empirical clean FPR) and "
                 "the mean +/- std recall across subjects. Answers whether the detector just "
                 "catches some users better than others.",
        'weights': str(weights),
        'x_axis': {'name': 'empirical clean FPR (grid) / expected FPR (aggregate)',
                   'range': [0, 1]},
        'y_axis': {'name': 'recall', 'range': [0, 1]},
        'measured_on': {
            'subjects': order,
            'note': "every subject scored on the given model; if --weights is an all-users "
                    "teacher then every subject was trained on, so this is the population "
                    "spread of per-subject detectability, not a generalization number."},
        'highlight': sorted(highlight),
        'aggregate': {'mean_auc': float(np.mean(aucs)), 'std_auc': float(np.std(aucs)),
                      'min_auc': float(np.min(aucs)), 'max_auc': float(np.max(aucs))},
        'per_subject': per_subject,
        'source': {'reproducible': True},
    })
