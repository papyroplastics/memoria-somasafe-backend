"""
Calibrate the autoencoder anomaly detector for label distillation: pick the per-score
budgets — the only globally-relevant output. A budget is the share of clean windows its
score may fire on (the quantile level a client will threshold at), chosen independently
on the labeled global data as the level that maximizes the score's Youden's J (mixed-set
recall minus clean FPR), capped at --max-budget. Writes only the budgets to
results/<model>/reports/; distill_labels.py turns them into per-subject thresholds +
labels and reports the detector's metrics.
"""


import argparse
import json

import numpy as np

from common.config import MODELS_DIR, DATASETS_DIR
from ml.model_list import MODELS
from ml.metrics import classification_report
from .common.post_train import get_report_dir, CALIBRATION_REPORT
from .common.autoencoders import load_autoencoder
from .common.scoring import (
    SCORE_NAMES, clean_threshold, score_dir_by_subject, score_mixed_by_subject,
    load_mixed_truth,
)

# Candidate per-score budgets (the share of clean windows a score may fire on = the
# quantile level its threshold sits at). Each score's budget is picked independently,
# so 0.0 (threshold at the subject's clean max, fires only above anything seen clean)
# is where a score with no useful signal lands on its own.
BUDGET_GRID = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)


def calibrate_budgets(clean: dict[str, dict[str, np.ndarray]],
                      mixed: dict[str, dict[str, np.ndarray]],
                      truth: dict[str, np.ndarray], max_budget: float,
                      grid=BUDGET_GRID) -> dict[str, float]:
    pooled_truth = np.concatenate([truth[sid] for sid in mixed])
    budgets: dict[str, float] = {}
    for n in SCORE_NAMES:
        best = None
        for b in grid:
            if b > max_budget:
                continue
            thr = {sid: clean_threshold(clean[sid][n], b) for sid in clean}
            recall = classification_report(
                np.concatenate([mixed[sid][n] > thr[sid] for sid in mixed]),
                pooled_truth)['recall']
            fpr = float(np.concatenate(
                [clean[sid][n] > thr[sid] for sid in clean]).mean())
            j = recall - fpr
            if best is None or j > best[0]:
                best = (j, b)
        budgets[n] = best[1]
    return budgets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to calibrate')
    parser.add_argument('--max-budget', type=float, default=0.03,
                        help='Per-score clean-window false-positive-rate ceiling the '
                             'budget search may reach (default: 0.03). Each score is '
                             'calibrated independently up to this cap; the combined FPR '
                             'that results is reported by distill_labels, not bounded here.')
    args = parser.parse_args()

    trainer = load_autoencoder(args.model)

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

    print("Calibrating per-score budgets...")
    budgets = calibrate_budgets(clean, mixed, truth, args.max_budget)

    print(f"\nbudgets (per-score clean-FPR share, independent; per-score cap {args.max_budget:.1%}):  "
          + "  ".join(f"{n}={budgets[n]:.4f}" for n in SCORE_NAMES))

    results = {'model': args.model, 'max_budget': args.max_budget, 'budgets': budgets}
    report_dir = get_report_dir(MODELS_DIR / args.model)
    report_path = report_dir / CALIBRATION_REPORT
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote budgets to {report_path}")
