"""Calibrate the detector's expected FPR (max Youden's J on the training subjects) and plot
the FPR sweep + ROC on the held-out subjects (report Sec. 5.4): two figures + the sweep
table under ``results/<model>/calibrate_fpr/``. ``--global-f`` swaps the per-subject
threshold for a single pooled one (population-level operating point).
"""

import argparse

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.model_list import MODELS
from ml.preprocessing import CLEAN_SUBDIR, MIXED_SUBDIR
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights, trainable_path

from ..common.plots import line_plot
from ..common.reports import get_report_dir, read_subject_split, write_metrics_csv, write_yaml
from ..common.scoring import (
    calibrate_expected_fpr, sweep_expected_fpr, subject_thresholds, global_thresholds,
    score_dir_by_subject, load_mixed_truth)


def build_grid(expected_fpr: float, step: float) -> list[float]:
    grid = set(np.round(np.arange(0.0, 1.0 + step / 2, step), 4).tolist())
    grid.add(round(expected_fpr, 4))
    return sorted(grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to calibrate')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Spacing of the expected-FPR sweep plotted on the held-out '
                             'subjects (default: 0.05)')
    parser.add_argument('--global-f', action='store_true',
                        help='Threshold with a single pooled clean quantile instead of a '
                             "per-subject one (each subject's clean FPR then drifts off f)")
    parser.add_argument('--tag', default=None,
                        help='Tag of the train.py run to calibrate (default: the canonical '
                             'untagged run). Selects both trainable_<tag>.tflite and the '
                             'normal_<tag>/federated_<tag> run.yaml it was trained with.')
    args = parser.parse_args()

    thresholds_fn = global_thresholds if args.global_f else subject_thresholds
    mode = 'global' if args.global_f else 'per-subject'

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    weights = trainable_path(MODELS_DIR / args.model, args.tag)
    trainer.model.restore(load_trainable_weights(weights))
    assert isinstance(trainer, AutoencoderTrainer)

    train_ids, held_out = read_subject_split(args.model, ('normal', 'federated'), args.tag)
    train, held = set(train_ids), set(held_out)

    print(f"Calibrating the expected FPR ({mode} threshold) on the {len(train_ids)} "
          f"training subjects: {', '.join(train_ids)}")
    truth_tr = load_mixed_truth(DATASETS_DIR)
    clean_tr = score_dir_by_subject(trainer, DATASETS_DIR / CLEAN_SUBDIR, subjects=train)
    mixed_tr = score_dir_by_subject(trainer, DATASETS_DIR / MIXED_SUBDIR, subjects=train)
    expected_fpr = calibrate_expected_fpr(clean_tr, mixed_tr, truth_tr, thresholds_fn=thresholds_fn)
    print(f"expected_fpr = {expected_fpr:.4f}")

    print(f"\nSweeping the FPR curve on the {len(held_out)} held-out subjects: "
          f"{', '.join(held_out)}")
    truth = load_mixed_truth(DATASETS_DIR)
    clean = score_dir_by_subject(trainer, DATASETS_DIR / CLEAN_SUBDIR, subjects=held)
    mixed = score_dir_by_subject(trainer, DATASETS_DIR / MIXED_SUBDIR, subjects=held)

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

    report_dir = get_report_dir(args.model, 'calibrate_fpr')
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

    threshold_desc = ("single global threshold: the 1-f quantile of all subjects' pooled "
                      "clean scores, applied unchanged to everyone (per-subject clean FPR "
                      "then drifts off f with each subject's own error scale)"
                      if args.global_f else
                      "per-subject threshold: the 1-f quantile of each subject's own clean "
                      "scores, so every subject's clean FPR is f by construction")

    write_metrics_csv(sweep, report_dir, 'calibration.csv')
    write_yaml(report_dir / 'calibration.yaml', {
        'shows': "Detector calibration sweep: how recall and the empirical clean "
                 "false-positive rate trade off as the expected FPR varies, and the "
                 "operating point selected from it.",
        'threshold': threshold_desc,
        'x_axis': {'name': 'expected FPR', 'range': [min(levels), max(levels)]},
        'y_axis': {'name': 'rate', 'range': [0, 1]},
        'measured_on': {
            'calibration_subjects': train_ids,
            'sweep_subjects': held_out,
            'note': "the operating point is selected on the training subjects; the sweep "
                    "plotted here is evaluated on the held-out subjects, so the numbers "
                    "are generalization to an unseen user."},
        'selection': {
            'criterion': "maximum Youden's J (recall - clean FPR), found by a dense "
                         "grid argmax (J is not unimodal, so a scan-then-ternary search "
                         "overshoots the peak and drifts across its noisy plateau)",
            'why': "J is built from two rates each conditioned on a single class, so it "
                   "is independent of the anomaly prevalence of the set it is measured "
                   "on; precision (and therefore F1) mixes the classes and inherits that "
                   "prevalence, so an F1-selected threshold would not transfer to a "
                   "deployment whose prevalence is unknown and subject-varying.",
            'expected_fpr': expected_fpr,
        },
        'sweep': sweep,
        'headline': chosen,
        'caveats': ["precision and F1 are reported at each level but are "
                    "prevalence-dependent: the mixed set is ~50% anomalous by "
                    "construction, and a real deployment's far lower rate would make "
                    "precision worse than shown"],
        'source': {'reproducible': True},
    })
    write_yaml(report_dir / 'roc.yaml', {
        'shows': "The detector's ROC curve on the held-out subjects: recall against the "
                 "empirical clean false-positive rate as the expected FPR sweeps from 0 to "
                 "1, with the selected operating point marked. See calibration.yaml/csv for "
                 "the full sweep (including precision/F1/Youden's J per level).",
        'threshold': threshold_desc,
        'x_axis': {'name': 'empirical clean FPR', 'range': [0, 1]},
        'y_axis': {'name': 'recall', 'range': [0, 1]},
        'measured_on': {'subjects': held_out,
                        'note': 'held-out subjects — generalization to an unseen user'},
        'headline': {'expected_fpr': expected_fpr, 'clean_fpr': chosen['clean_fpr'],
                     'recall': chosen['recall']},
        'source': {'reproducible': True},
    })
