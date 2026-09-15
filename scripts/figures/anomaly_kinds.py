"""Scores every anomaly kind in isolation at one fixed operating point, reporting recall
at the calibrated expected FPR and the threshold-free AUC against clean windows, to show
where a detector's aggregate numbers come from and which kinds it is structurally blind
to."""

import argparse

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.dataset_list import BASES
from ml.model_list import MODELS
from ml.saving import load_weights, weights_path
from ml.sources.dalia import ANOMALY_KINDS, CLEAN, MIXED

from ..common.plots import bar_plot
from ..common.reports import get_report_dir, read_subject_split, write_metrics_csv, write_yaml
from ..common.scoring import (
    calibrate_expected_fpr, mixed_truth, score_subjects, subject_thresholds, variant_sources,
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
    parser.add_argument('--dataset', choices=sorted(BASES), default='ppg-dalia-low',
                        help='Dataset to score on')
    args = parser.parse_args()

    model = MODELS[args.model].build_model(DATASETS_DIR)
    sources = variant_sources(model, args.dataset, DATASETS_DIR)
    weights = weights_path(MODELS_DIR / args.model, args.tag)
    model.restore(load_weights(weights))

    train_ids, held_out = read_subject_split(args.model, ('normal', 'federated'), args.tag)
    train, held = set(train_ids), set(held_out)

    print(f"Scoring {BASES[args.dataset].name}")
    expected_fpr = args.expected_fpr
    if expected_fpr is None:
        print(f"Calibrating the expected FPR on the {len(train_ids)} training subjects...")
        expected_fpr = calibrate_expected_fpr(
            score_subjects(model, sources[CLEAN], subjects=train),
            score_subjects(model, sources[MIXED], subjects=train),
            mixed_truth(sources[MIXED], subjects=train))
    print(f"expected_fpr = {expected_fpr:.4f}")

    print(f"Scoring every kind on the {len(held_out)} held-out subjects: "
          f"{', '.join(held_out)}")
    clean = score_subjects(model, sources[CLEAN], subjects=held)
    thresholds = subject_thresholds(clean, expected_fpr)

    rows = []
    for kind in ANOMALY_KINDS:
        scores = score_subjects(model, sources[kind], subjects=held)
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
        'dataset': args.dataset,
        'calibration_subjects': train_ids,
        'eval_subjects': held_out,
        'expected_fpr': expected_fpr,
        'clean_fpr': clean_fpr,
    })
