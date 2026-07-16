"""
Calibrate the autoencoder anomaly detector for label distillation: pick the expected FPR —
the only globally-relevant output, and the only thing that reads the synthetic labels. It
is the rate at which the detector fires on clean signal, and equivalently the quantile
level a client thresholds its own clean baseline at: the server calibrates the rate, the
client computes the threshold. It is chosen as the level maximizing the detector's Youden's
J (mixed-set recall minus empirical clean false-positive rate) on the labeled global data.
Writes only the expected FPR to results/<model>/; distill_labels.py turns it into
per-subject thresholds + labels and distill_eval.py reports the detector's metrics.
"""


import argparse
import json
import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.model_list import MODELS
from ml.metrics import classification_report
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights

from ..common.reports import get_report_dir
from ..common.scoring import (
    CALIBRATION_REPORT, clean_threshold, score_dir_by_subject,
    score_mixed_by_subject, load_mixed_truth, DETECTOR)

# Candidate expected FPRs. Because each threshold is the (1 - f) quantile of the subject's
# *own* clean scores, that fraction of clean windows lies above it by definition — so
# J(f) = recall(f) - f and the grid just has to be wide enough to bracket the turn.
FPR_GRID = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1,
            0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def calibrate_expected_fpr(clean: dict[str, dict[str, np.ndarray]],
                           mixed: dict[str, dict[str, np.ndarray]],
                           truth: dict[str, np.ndarray],
                           grid=FPR_GRID) -> tuple[float, list[dict]]:
    """The expected FPR maximizing the detector's Youden's J, plus the whole sweep for the
    report.

    J rather than F1 because the mixed set is ~ANOMALY_PROB (50%) anomalous *by
    construction*: F1 depends on that prevalence and a synthetic 50% base rate is nothing
    a deployed detector would meet, while J = recall - FPR is prevalence-independent. F1
    and precision are recorded per level anyway, since the report shows the trade."""
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])
    sweep = []
    for f in grid:
        thr = {sid: clean_threshold(clean[sid][DETECTOR], f) for sid in clean}
        rep = classification_report(
            np.concatenate([mixed[sid][DETECTOR] > thr[sid] for sid in mixed]),
            pooled_truth)
        fpr = float(np.concatenate(
            [clean[sid][DETECTOR] > thr[sid] for sid in clean]).mean())
        sweep.append({'expected_fpr': f, 'recall': rep['recall'], 'precision': rep['precision'],
                      'f1': rep['f1'], 'clean_fpr': fpr, 'youden_j': rep['recall'] - fpr})
    best = max(sweep, key=lambda row: row['youden_j'])
    return best['expected_fpr'], sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to calibrate')
    args = parser.parse_args()

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    trainer.model.restore(load_trainable_weights(MODELS_DIR / args.model / 'trainable.tflite'))
    assert isinstance(trainer, AutoencoderTrainer)

    print("Scoring mixed-anomaly windows...")
    mixed = score_mixed_by_subject(trainer, DATASETS_DIR)
    truth = load_mixed_truth(DATASETS_DIR, mixed)

    print("Scoring clean windows...")
    clean = score_dir_by_subject(trainer, DATASETS_DIR, None)
    if not clean:
        raise SystemExit("no clean windows to set thresholds from.")
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; "
                         "cannot calibrate.")

    print("Calibrating the expected FPR...")
    expected_fpr, sweep = calibrate_expected_fpr(clean, mixed, truth)

    print(f"\n  {'exp_fpr':>8} {'recall':>8} {'precision':>10} {'f1':>8} {'clean_fpr':>10} {'youden_j':>9}")
    for row in sweep:
        mark = '  <-' if row['expected_fpr'] == expected_fpr else ''
        print(f"  {row['expected_fpr']:>8.4f} {row['recall']:>8.4f} {row['precision']:>10.4f} "
              f"{row['f1']:>8.4f} {row['clean_fpr']:>10.4f} {row['youden_j']:>9.4f}{mark}")
    print(f"\nexpected FPR = {expected_fpr:.4f}  (maximizes Youden's J; F1 is "
          f"prevalence-dependent and the mixed set is 50% anomalous by construction)")

    results = {'model': args.model, 'expected_fpr': expected_fpr, 'sweep': sweep}
    report_dir = get_report_dir(args.model)
    report_path = report_dir / CALIBRATION_REPORT
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote expected FPR to {report_path}")
