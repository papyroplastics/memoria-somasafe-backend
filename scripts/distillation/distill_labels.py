"""
Distill window labels from a trained autoencoder — the client-facing step. Touches only
data a real client would have: its own clean-signal baseline and the mixed signal to be
labeled (plus the features it computes on-device), never the ground-truth labels or the
per-anomaly datasets. Reads the global per-score budgets from distill_calibrate.py,
derives each subject's thresholds from its own clean windows, and emits a soft [0,1]
anomaly label per window — the clean-CDF rank past each score's threshold, max'd across
recon / spectral / rr, then temporally median-smoothed — into a datasets-shaped tree
(mixed-features/S*/ with distilled labels.npy + symlinked features) under results/<model>/
that train.py consumes via --dataset-dir. For the labeled diagnostics see distill_eval.py.
"""


import argparse
from pathlib import Path

import numpy as np

from common.config import RESULTS_DIR, DATASETS_DIR
from ml.data import MIXED_FEATURE_SUBDIR, FEATURE_STATS_FILE, BVP_WINDOW, WINDOW_SECONDS
from ml.model_list import MODELS
from ..common.autoencoders import load_autoencoder
from ..common.scoring import (
    SCORE_NAMES, subject_thresholds, soft_score, median3,
    score_dir_by_subject, score_mixed_by_subject,
)
from ..common.post_train import load_budgets


def relink(link: Path, target: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = target.resolve().relative_to(link.parent.resolve(), walk_up=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to distill from')
    parser.add_argument('--out-subdir', default='distilled-labels',
                        help='Subdirectory of results/<model>/ for the labels (default: distilled-labels)')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    result_dir = RESULTS_DIR / args.model

    budgets = load_budgets(args.model)
    # batch_size=1 so every window is scored (no batch remainder dropped): distilled
    # labels line up 1:1 with the feature windows, and each subject's thresholds are set
    # from its full clean set — the same thing an on-device client does.
    trainer = load_autoencoder(args.model, batch_size=1)

    window = trainer.window_size
    if window != BVP_WINDOW:
        raise SystemExit(
            f"model window ({window} samples) does not match the {WINDOW_SECONDS}s feature "
            f"window ({BVP_WINDOW} samples) used to build mixed-features; the autoencoder "
            f"would produce a mismatched number of labels. Align the model's window size.")

    print("Scoring mixed-anomaly windows...")
    mixed = score_mixed_by_subject(trainer, data_dir)
    print("Scoring clean windows (sets each subject's thresholds)...")
    clean = score_dir_by_subject(trainer, data_dir, None)
    missing = set(mixed) - set(clean)
    if missing:
        raise SystemExit(f"subjects {sorted(missing)} lack clean windows; "
                         "cannot derive per-subject thresholds.")

    thresholds = subject_thresholds(clean, budgets)

    # Soft labels: clean-CDF rank past each subject's threshold, max over scores, then a
    # size-1 temporal median filter. Mirror the feature dataset's structure under out_dir
    # so it can be passed to train.py as --dataset-dir; only labels.npy is written, the
    # feature arrays and global stats are symlinked back to the real dataset.
    out_dir = result_dir / args.out_subdir
    out_feature_dir = out_dir / MIXED_FEATURE_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR
    print("Writing soft labels (budgets "
          + ", ".join(f"{n}={budgets[n]:.4f}" for n in SCORE_NAMES) + "):")
    for sid in mixed:
        soft = median3(soft_score(mixed[sid], clean[sid], budgets))
        save_dir = out_feature_dir / sid
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'labels.npy', soft.reshape(-1, 1).astype(np.float32))
        relink(save_dir / 'features.npy', feature_dir / sid / 'features.npy')
        print(f"  {sid}: {len(soft)} windows, mean soft label {soft.mean():.3f}, "
              f"hard rate {(soft > 0).mean():.1%}")
    relink(out_feature_dir / FEATURE_STATS_FILE, feature_dir / FEATURE_STATS_FILE)
    np.save(out_dir / 'budgets.npy',
            np.array([budgets[n] for n in SCORE_NAMES], dtype=np.float32))
    print(f"Wrote distilled-label dataset for {len(mixed)} subjects to {out_dir}/")
