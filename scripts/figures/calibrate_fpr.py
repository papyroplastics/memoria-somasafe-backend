"""Calibrate the detector's expected FPR (max Youden's J on the training subjects) and plot
the FPR sweep + ROC on the held-out ones (report Sec. 5.4): two figures + the sweep table
under ``results/<model>/calibrate_fpr/<dataset>/``."""

import argparse

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.dataset_list import BASES
from ml.model_list import MODELS
from ml.saving import load_weights, weights_path
from ml.sources.dalia import CLEAN, MIXED

from ..common.plots import line_plot
from ..common.reports import get_report_dir, read_subject_split, write_metrics_csv, write_yaml
from ..common.scoring import (
    calibrate_expected_fpr, sweep_expected_fpr, subject_thresholds, global_thresholds,
    score_subjects, mixed_truth, variant_sources)


def build_grid(expected_fpr: float, step: float) -> list[float]:
    grid = set(np.round(np.arange(0.0, 1.0 + step / 2, step), 4).tolist())
    grid.add(round(expected_fpr, 4))
    return sorted(grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to calibrate')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Spacing of the expected-FPR sweep')
    parser.add_argument('--global-f', action='store_true',
                        help='Threshold with a single pooled clean quantile instead of a '
                             "per-subject one (each subject's clean FPR then drifts off f)")
    parser.add_argument('--tag', default=None,
                        help='Tag of the train.py run to calibrate, selecting both its '
                             'weights and the run.yaml it was trained with')
    parser.add_argument('--dataset', choices=sorted(BASES), default='ppg-dalia',
                        help='Dataset to calibrate and sweep on; ppg-dalia-low keeps only '
                             'the low-activity windows the model was trained on')
    args = parser.parse_args()

    thresholds_fn = global_thresholds if args.global_f else subject_thresholds
    mode = 'global' if args.global_f else 'per-subject'

    model = MODELS[args.model].build_model(DATASETS_DIR)
    sources = variant_sources(model, args.dataset, DATASETS_DIR, variants=(CLEAN, MIXED))
    weights = weights_path(MODELS_DIR / args.model, args.tag)
    model.restore(load_weights(weights))

    train_ids, held_out = read_subject_split(args.model, ('normal', 'federated'), args.tag)
    train, held = set(train_ids), set(held_out)

    print(f"Scoring {BASES[args.dataset].name}")
    print(f"Calibrating the expected FPR ({mode} threshold) on the {len(train_ids)} "
          f"training subjects: {', '.join(train_ids)}")
    truth_tr = mixed_truth(sources[MIXED], subjects=train)
    clean_tr = score_subjects(model, sources[CLEAN], subjects=train)
    mixed_tr = score_subjects(model, sources[MIXED], subjects=train)
    expected_fpr = calibrate_expected_fpr(clean_tr, mixed_tr, truth_tr, thresholds_fn=thresholds_fn)
    print(f"expected_fpr = {expected_fpr:.4f}")

    print(f"\nSweeping the FPR curve on the {len(held_out)} held-out subjects: "
          f"{', '.join(held_out)}")
    truth = mixed_truth(sources[MIXED], subjects=held)
    clean = score_subjects(model, sources[CLEAN], subjects=held)
    mixed = score_subjects(model, sources[MIXED], subjects=held)

    grid = build_grid(expected_fpr, args.step)
    sweep = sweep_expected_fpr(clean, mixed, truth, grid, thresholds_fn)
    chosen = next(row for row in sweep if row['expected_fpr'] == round(expected_fpr, 4))

    print(f"\n  {'exp_fpr':>8} {'recall':>8} {'precision':>10} {'f1':>8} {'clean_fpr':>10} {'youden_j':>9}")
    for row in sweep:
        mark = '  <-' if row is chosen else ''
        print(f"  {row['expected_fpr']:>8.4f} {row['recall']:>8.4f} {row['precision']:>10.4f} "
              f"{row['f1']:>8.4f} {row['clean_fpr']:>10.4f} {row['youden_j']:>9.4f}{mark}")
    print(f"\nexpected FPR = {expected_fpr:.4f}  (maximizes Youden's J on the training "
          f"subjects; F1 is prevalence-dependent and the mixed set is 50% anomalous by "
          f"construction)")

    report_dir = get_report_dir(args.model, f'calibrate_fpr/{args.dataset}')
    levels = [row['expected_fpr'] for row in sweep]

    line_plot(report_dir / 'calibration.png', levels,
              {'recall (mixed set)': [row['recall'] for row in sweep],
               'empirical clean FPR': [row['clean_fpr'] for row in sweep],
               "Youden's J = recall - FPR": [row['youden_j'] for row in sweep]},
              'expected FPR (calibrated clean false-positive rate)', 'rate',
              f'{args.model} — detector calibration, {mode} threshold (held-out subjects)',
              vline=(expected_fpr, f'selected expected FPR {expected_fpr:.4f}'))

    roc_order = sorted(range(len(sweep)), key=lambda i: sweep[i]['clean_fpr'])
    line_plot(report_dir / 'roc.png',
              [sweep[i]['clean_fpr'] for i in roc_order],
              {'recall': [sweep[i]['recall'] for i in roc_order]},
              'empirical clean FPR', 'recall',
              f'{args.model} — detector ROC, {mode} threshold (held-out subjects)',
              vline=(chosen['clean_fpr'],
                     f"selected operating point (FPR={chosen['clean_fpr']:.4f})"),
              diagonal=True)

    write_metrics_csv(sweep, report_dir, 'calibration.csv')
    write_yaml(report_dir / 'calibration.yaml', {
        'dataset': args.dataset,
        'global_f': args.global_f,
        'calibration_subjects': train_ids,
        'sweep_subjects': held_out,
        'expected_fpr': expected_fpr,
    })
    write_yaml(report_dir / 'roc.yaml', {
        'dataset': args.dataset,
        'global_f': args.global_f,
        'subjects': held_out,
        'expected_fpr': expected_fpr,
        'clean_fpr': chosen['clean_fpr'],
        'recall': chosen['recall'],
    })
