"""Per-subject ROC of the reconstruction-error detector on one set of weights, to show
detectability varies by user (report Sec. 5.4): one ROC panel per subject and the mean +/-
std recall across them, under ``results/<model>/subject_roc/``."""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from common.config import DATASETS_DIR, MODELS_DIR
from ml.dataset_list import DATASETS
from ml.model_list import MODELS
from ml.saving import load_weights, weights_path
from ml.sources.dalia import CLEAN, MIXED

from ..common.plots import roc_grid
from ..common.reports import get_report_dir, write_yaml
from ..common.scoring import score_subjects, mixed_truth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to score')
    parser.add_argument('--tag', default=None,
                        help='Tag of the train.py run to score, e.g. an all-users '
                             "teacher's, which puts every subject on equal footing")
    parser.add_argument('--highlight', default='',
                        help='Comma-separated subject ids to draw in red')
    parser.add_argument('--global-f', action='store_true',
                        help='Threshold with a single pooled clean quantile instead of a '
                             "per-subject one (each subject's clean FPR then drifts off f)")
    parser.add_argument('--step', type=float, default=0.02,
                        help='Spacing of the FPR sweep')
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='ppg-dalia',
                        help='Dataset to score on; ppg-dalia-low keeps only the '
                             'low-activity windows the model was trained on')
    args = parser.parse_args()

    source = DATASETS[args.dataset].build(DATASETS_DIR)
    weights = weights_path(MODELS_DIR / args.model, args.tag)
    model = MODELS[args.model].build_model(DATASETS_DIR)
    model.restore(load_weights(weights))

    print(f"Scoring {DATASETS[args.dataset].name}")
    truth = mixed_truth(source)
    clean = score_subjects(model, source, CLEAN)
    mixed = score_subjects(model, source, MIXED)

    order = [sid for sid in clean if sid in mixed and sid in truth]
    highlight = {f'S{int(i)}' for i in args.highlight.split(',') if i.strip()}
    mode = 'global' if args.global_f else 'per-subject'

    fine = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 4)
    gthr = (np.quantile(np.concatenate([clean[sid] for sid in order]), 1.0 - fine)
            if args.global_f else None)
    curves, per_subject, recalls = {}, {}, []
    for sid in order:
        c, m, t = clean[sid], mixed[sid], truth[sid]
        thr = gthr if args.global_f else np.quantile(c, 1.0 - fine)
        fpr = (c[:, None] > thr).mean(axis=0)
        anom = m[t == 1]
        recall = (anom[:, None] > thr).mean(axis=0) if len(anom) else np.full_like(fpr, np.nan)
        curves[sid] = (fpr.tolist(), recall.tolist())
        recalls.append(recall)
        per_subject[sid] = {'auc': float(np.trapezoid(recall, fpr)),
                            'anomalous_windows': int((t == 1).sum())}

    report_dir = get_report_dir(args.model, f'subject_roc/{args.dataset}')
    roc_grid(report_dir / 'roc_by_subject.png', order, curves, highlight,
             'empirical clean FPR', 'recall',
             f'{args.model} — per-subject ROC, {mode} threshold ({weights.name})')

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
    ax.set_title(f'{args.model} — recall vs. expected FPR, {mode} threshold, '
                 f'mean ± std over {len(order)} subjects')
    ax.legend()
    fig.savefig(report_dir / 'roc_aggregate.png')
    plt.close(fig)
    print(f"saved plot to {report_dir / 'roc_aggregate.png'}")

    aucs = [s['auc'] for s in per_subject.values()]
    write_yaml(report_dir / 'subject_roc.yaml', {
        'dataset': {'key': args.dataset, 'name': DATASETS[args.dataset].name},
        'shows': "Per-subject detectability of the reconstruction-error detector on one set "
                 "of weights: each subject's ROC (recall vs. its own empirical clean FPR) and "
                 "the mean +/- std recall across subjects. Answers whether the detector just "
                 "catches some users better than others.",
        'weights': str(weights),
        'threshold': ("single global threshold (the 1-f quantile of all subjects' pooled "
                      "clean scores) — each subject's clean FPR drifts off f by its own "
                      "error scale" if args.global_f else
                      "per-subject threshold (the 1-f quantile of each subject's own clean "
                      "scores) — each subject's clean FPR is f by construction"),
        'x_axis': {'name': 'empirical clean FPR (grid) / expected FPR (aggregate)',
                   'range': [0, 1]},
        'y_axis': {'name': 'recall', 'range': [0, 1]},
        'measured_on': {
            'subjects': order,
            'note': "every subject scored on the given model; if --tag is an all-users "
                    "teacher then every subject was trained on, so this is the population "
                    "spread of per-subject detectability, not a generalization number."},
        'highlight': sorted(highlight),
        'aggregate': {'mean_auc': float(np.mean(aucs)), 'std_auc': float(np.std(aucs)),
                      'min_auc': float(np.min(aucs)), 'max_auc': float(np.max(aucs))},
        'per_subject': per_subject,
        'source': {'reproducible': True},
    })
