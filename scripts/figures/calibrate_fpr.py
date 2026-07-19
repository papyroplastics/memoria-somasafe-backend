"""
Calibrate the autoencoder detector's expected FPR and produce the whole calibration story
for the report (Sec. 5.4): two figures plus the raw sweep table.

The operating point (the expected FPR maximizing Youden's J) is picked on the model's
**training** subjects via a dense grid argmax (cheap; see
``scripts.common.scoring.calibrate_expected_fpr``) — the same subjects and the same value
``anomaly_detection.py`` and ``knowledge_distillation.py`` each re-pick internally. The whole
FPR sweep plotted here, though, is evaluated on the **held-out** subjects instead: each
subject's threshold is defined as a quantile of that very subject's own clean scores, so a
sweep measured on the calibration subjects would show the empirical clean FPR tracking the
expected FPR almost exactly by construction — not a generalization claim. Scoring only the
held-out subjects is both cheaper (2 subjects vs. the training split) and the honest number:
recall/FPR as they would land for an unseen user.

Two figures come out of one run, both under ``results/<model>/calibrate_fpr/``:

    calibration.png   recall / empirical clean FPR / Youden's J vs. *expected* FPR
    roc.png            recall vs. *empirical* clean FPR — the actual ROC curve

Nothing else imports this script; anomaly_detection.py and knowledge_distillation.py each
call ``calibrate_expected_fpr`` themselves, since the search is cheap.
"""

import argparse

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.model_list import MODELS
from ml.preprocessing import CLEAN_SUBDIR, MIXED_SUBDIR
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights

from ..common.plots import line_plot
from ..common.reports import get_report_dir, read_subject_split, write_metrics_csv, write_yaml
from ..common.scoring import (
    calibrate_expected_fpr, sweep_expected_fpr, score_dir_by_subject, load_mixed_truth)


def build_grid(expected_fpr: float, step: float) -> list[float]:
    """0..1 in ``step`` increments, plus the selected point itself so it is always an exact
    row in the sweep rather than merely near one."""
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
    args = parser.parse_args()

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    trainer.model.restore(load_trainable_weights(MODELS_DIR / args.model / 'trainable.tflite'))
    assert isinstance(trainer, AutoencoderTrainer)

    train_ids, held_out = read_subject_split(args.model, ('normal', 'federated'))
    train, held = set(train_ids), set(held_out)

    print(f"Calibrating the expected FPR on the {len(train_ids)} training subjects: "
          f"{', '.join(train_ids)}")
    mixed_tr = score_dir_by_subject(trainer, DATASETS_DIR / MIXED_SUBDIR, subjects=train)
    truth_tr = load_mixed_truth(DATASETS_DIR, mixed_tr)
    clean_tr = score_dir_by_subject(trainer, DATASETS_DIR / CLEAN_SUBDIR, subjects=train)
    expected_fpr = calibrate_expected_fpr(clean_tr, mixed_tr, truth_tr)
    print(f"expected_fpr = {expected_fpr:.4f}")

    print(f"\nSweeping the FPR curve on the {len(held_out)} held-out subjects: "
          f"{', '.join(held_out)}")
    mixed = score_dir_by_subject(trainer, DATASETS_DIR / MIXED_SUBDIR, subjects=held)
    truth = load_mixed_truth(DATASETS_DIR, mixed)
    clean = score_dir_by_subject(trainer, DATASETS_DIR / CLEAN_SUBDIR, subjects=held)
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; cannot sweep.")

    grid = build_grid(expected_fpr, args.step)
    sweep = sweep_expected_fpr(clean, mixed, truth, grid)
    sweep_tr = sweep_expected_fpr(clean_tr, mixed_tr, truth_tr, grid)
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
               "Youden's J = recall - FPR": [row['youden_j'] for row in sweep],
               "Youden's J (calibration subjects)": [row['youden_j'] for row in sweep_tr]},
              'expected FPR (calibrated clean false-positive rate)', 'rate',
              f'{args.model} — detector calibration (held-out subjects)',
              vline=(expected_fpr, f'selected expected FPR {expected_fpr:.4f}'))

    roc_order = sorted(range(len(sweep)), key=lambda i: sweep[i]['clean_fpr'])
    line_plot(report_dir / 'roc.png',
              [sweep[i]['clean_fpr'] for i in roc_order],
              {'recall': [sweep[i]['recall'] for i in roc_order]},
              'empirical clean FPR', 'recall',
              f'{args.model} — detector ROC (held-out subjects)',
              vline=(chosen['clean_fpr'],
                     f"selected operating point (FPR={chosen['clean_fpr']:.4f})"),
              diagonal=True)

    write_metrics_csv(sweep, report_dir, 'calibration.csv')
    write_yaml(report_dir / 'calibration.yaml', {
        'shows': "Detector calibration sweep: how recall and the empirical clean "
                 "false-positive rate trade off as the expected FPR varies, and the "
                 "operating point selected from it. The expected FPR is the rate at which "
                 "the detector fires on clean signal; a client turns it into a threshold "
                 "at the 1-f quantile of its own clean reconstruction errors, so that "
                 "fraction of clean windows lies above the threshold by definition — the "
                 "parameter is the false-alarm rate, not a proxy for it.",
        'x_axis': {'name': 'expected FPR', 'range': [min(levels), max(levels)]},
        'y_axis': {'name': 'rate', 'range': [0, 1]},
        'measured_on': {
            'calibration_subjects': train_ids,
            'sweep_subjects': held_out,
            'note': "the operating point is selected on the training subjects; the sweep "
                    "plotted here is evaluated on the held-out subjects, so the empirical "
                    "clean FPR is a generalization number rather than the tautological "
                    "match a calibration-subject sweep would show (each threshold is "
                    "already a quantile of its own subject's clean scores)."},
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
        'x_axis': {'name': 'empirical clean FPR', 'range': [0, 1]},
        'y_axis': {'name': 'recall', 'range': [0, 1]},
        'measured_on': {'subjects': held_out,
                         'note': 'held-out subjects — generalization to an unseen user, '
                                 'unlike the calibration-subject sweep which would show the '
                                 'empirical FPR tracking the expected FPR by construction'},
        'headline': {'expected_fpr': expected_fpr, 'clean_fpr': chosen['clean_fpr'],
                     'recall': chosen['recall']},
        'source': {'reproducible': True},
    })
