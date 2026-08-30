"""Scores every anomaly kind in isolation at one fixed operating point, reporting recall
at the calibrated expected FPR and the threshold-free AUC against clean windows, to show
where a detector's aggregate numbers come from and which kinds it is structurally blind
to."""

import argparse

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.dataset_list import DATASETS
from ml.model_list import MODELS
from ml.preprocessing import ANOMALY_KINDS
from ml.saving import load_weights, weights_path
from ml.sources.dalia import CLEAN, MIXED

from ..common.plots import bar_plot
from ..common.reports import get_report_dir, read_subject_split, write_metrics_csv, write_yaml
from ..common.scoring import (
    calibrate_expected_fpr, mixed_truth, score_subjects, subject_thresholds,
)


def roc_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """The Mann-Whitney U statistic, i.e. the area under the ROC this pair of score sets would trace."""
    scores = np.concatenate([negative, positive])
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    values = scores[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            ranks[order[start:i]] = (start + i + 1) / 2.0
            start = i

    n_pos, n_neg = len(positive), len(negative)
    if not n_pos or not n_neg:
        return float('nan')
    return float((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to score')
    parser.add_argument('--tag', default=None, help='Tag of the train.py run to score')
    parser.add_argument('--expected-fpr', type=float, default=None,
                        help='Operating point to score at')
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='ppg-dalia-low',
                        help='Dataset to score on')
    args = parser.parse_args()

    source = DATASETS[args.dataset].build(DATASETS_DIR)
    model = MODELS[args.model].build_model(DATASETS_DIR)
    weights = weights_path(MODELS_DIR / args.model, args.tag)
    model.restore(load_weights(weights))

    train_ids, held_out = read_subject_split(args.model, ('normal', 'federated'), args.tag)
    train, held = set(train_ids), set(held_out)

    print(f"Scoring {DATASETS[args.dataset].name}")
    expected_fpr = args.expected_fpr
    if expected_fpr is None:
        print(f"Calibrating the expected FPR on the {len(train_ids)} training subjects...")
        expected_fpr = calibrate_expected_fpr(
            score_subjects(model, source, CLEAN, subjects=train),
            score_subjects(model, source, MIXED, subjects=train),
            mixed_truth(source, subjects=train))
    print(f"expected_fpr = {expected_fpr:.4f}")

    print(f"Scoring every kind on the {len(held_out)} held-out subjects: "
          f"{', '.join(held_out)}")
    clean = score_subjects(model, source, CLEAN, subjects=held)
    thresholds = subject_thresholds(clean, expected_fpr)

    rows = []
    for kind in ANOMALY_KINDS:
        scores = score_subjects(model, source, kind, subjects=held)
        flags = np.concatenate([scores[sid] > thresholds[sid] for sid in scores])
        rows.append({
            'kind': kind,
            'windows': int(len(flags)),
            'recall': float(flags.mean()),
            'auc': roc_auc(np.concatenate([clean[sid] for sid in clean]),
                           np.concatenate([scores[sid] for sid in scores])),
        })

    clean_fpr = float(np.concatenate(
        [clean[sid] > thresholds[sid] for sid in clean]).mean())

    print(f"\n  {'kind':<9} {'recall':>8} {'AUC':>8}   verdict")
    for row in rows:
        verdict = ('inverted — scored as more normal than clean signal'
                   if row['auc'] < 0.5 else
                   'weak' if row['auc'] < 0.65 else 'detected')
        print(f"  {row['kind']:<9} {row['recall']:>8.4f} {row['auc']:>8.4f}   {verdict}")
    print(f"\nclean FPR at this threshold: {clean_fpr:.4f} "
          f"(a kind whose recall sits below it is worse than flagging at random)")

    report_dir = get_report_dir(args.model, f'anomaly_kinds/{args.dataset}')
    bar_plot(report_dir / 'anomaly_kinds.png',
             [row['kind'] for row in rows],
             {'recall at the operating point': [row['recall'] for row in rows],
              'AUC vs. clean windows': [row['auc'] for row in rows]},
             'anomaly kind', 'rate',
             f'{args.model} — detectability by anomaly kind (held-out subjects)',
             hlines=[(clean_fpr, f'clean FPR {clean_fpr:.3f}'),
                     (0.5, 'AUC 0.5 — no separation from clean')])

    write_metrics_csv(rows, report_dir, 'anomaly_kinds.csv')
    write_yaml(report_dir / 'anomaly_kinds.yaml', {
        'dataset': {'key': args.dataset, 'name': DATASETS[args.dataset].name},
        'shows': "Per-anomaly-kind detectability at one fixed operating point: recall on "
                 "each kind's fully-anomalous set, and the threshold-free AUC of its "
                 "scores against the same subjects' clean windows. Separates the kinds a "
                 "detector catches from the ones it is structurally blind to, which the "
                 "aggregate mixed-set metrics average together.",
        'threshold': "per-subject threshold: the 1-f quantile of each subject's own clean "
                     "scores, the same thresholds calibrate_fpr.py sweeps",
        'x_axis': {'name': 'anomaly kind'},
        'y_axis': {'name': 'rate', 'range': [0, 1]},
        'measured_on': {
            'calibration_subjects': train_ids,
            'eval_subjects': held_out,
            'note': "each kind is scored on its own fully-anomalous copy of the held-out "
                    "subjects, never on the mix, so no kind's score is diluted by the "
                    "others."},
        'selection': {'expected_fpr': expected_fpr, 'clean_fpr': clean_fpr},
        'reading': "AUC below 0.5 means the score is inverted for that kind: the detector "
                   "ranks those windows as more normal than clean signal, so no threshold "
                   "on this score can catch them and the aggregate metrics hide that "
                   "behind the kinds it does catch. A recall below the clean FPR on the "
                   "same row is the same finding read at the chosen operating point.",
        'per_kind': rows,
        'source': {'reproducible': True},
    })
