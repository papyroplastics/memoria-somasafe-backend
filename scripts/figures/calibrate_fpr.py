"""
Sweep the autoencoder detector's expected FPR and report the whole table (report Sec. 5.4):
for each candidate FPR, recall / precision / F1 / empirical clean FPR / Youden's J, with the
level that maximizes J marked. This exists to justify calibrating on J rather than F1 — the
table shows F1 is prevalence-dependent (the mixed set is 50% anomalous by construction) while
J = recall - FPR is not. It is standalone: anomaly_detection.py and knowledge_distillation.py
each re-pick the FPR internally, so nothing imports this; it only writes the sweep for the
report figure (plot_calibration.py) and the table.

Calibrates on the model's training subjects (the split read from its run manifest), matching
the operating point anomaly_detection.py evaluates at.
"""

import argparse
import json

import numpy as np

from common.config import DATASETS_DIR, MODELS_DIR
from ml.model_list import MODELS
from ml.models.common import AutoencoderTrainer
from ml.saving import load_trainable_weights

from ..common.reports import get_report_dir, read_eval_subjects
from ..common.scoring import (
    CALIBRATION_REPORT, calibrate_expected_fpr, score_dir_by_subject,
    score_mixed_by_subject, load_mixed_truth, split_subject_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to calibrate')
    args = parser.parse_args()

    trainer = MODELS[args.model].build_trainer(DATASETS_DIR)
    trainer.model.restore(load_trainable_weights(MODELS_DIR / args.model / 'trainable.tflite'))
    assert isinstance(trainer, AutoencoderTrainer)

    n_eval = read_eval_subjects(args.model, ('normal', 'federated'))
    train_ids, _ = split_subject_ids(DATASETS_DIR, n_eval)
    train = set(train_ids)
    print(f"Calibrating on the {len(train_ids)} training subjects: {', '.join(train_ids)}")

    print("Scoring mixed-anomaly windows...")
    mixed = score_mixed_by_subject(trainer, DATASETS_DIR, subjects=train)
    truth = load_mixed_truth(DATASETS_DIR, mixed)

    print("Scoring clean windows...")
    clean = score_dir_by_subject(trainer, DATASETS_DIR, None, subjects=train)
    if not clean:
        raise SystemExit("no clean windows to set thresholds from.")
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; cannot calibrate.")

    print("Sweeping the expected FPR...")
    expected_fpr, sweep = calibrate_expected_fpr(clean, mixed, truth)

    print(f"\n  {'exp_fpr':>8} {'recall':>8} {'precision':>10} {'f1':>8} {'clean_fpr':>10} {'youden_j':>9}")
    for row in sweep:
        mark = '  <-' if row['expected_fpr'] == expected_fpr else ''
        print(f"  {row['expected_fpr']:>8.4f} {row['recall']:>8.4f} {row['precision']:>10.4f} "
              f"{row['f1']:>8.4f} {row['clean_fpr']:>10.4f} {row['youden_j']:>9.4f}{mark}")
    print(f"\nexpected FPR = {expected_fpr:.4f}  (maximizes Youden's J; F1 is "
          f"prevalence-dependent and the mixed set is 50% anomalous by construction)")

    results = {'model': args.model, 'expected_fpr': expected_fpr,
               'calibration_subjects': train_ids, 'sweep': sweep}
    report_dir = get_report_dir(args.model)
    report_path = report_dir / CALIBRATION_REPORT
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote FPR sweep to {report_path}")
