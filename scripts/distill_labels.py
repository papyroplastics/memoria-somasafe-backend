import argparse
import json
from pathlib import Path

import numpy as np

from common.config import RESULTS_DIR, DATASETS_DIR
from ml.data import (
    CLEAN_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR, FEATURE_STATS_FILE,
    BVP_WINDOW, WINDOW_SECONDS, conditional_windows, get_sorted_paths
)
from ml.model_list import MODELS
from .common.autoencoders import load_autoencoder
from .common.scoring import SCORE_NAMES, score_windows, predict
from .common.post_train import get_report_dir, AE_TEST_REPORT


def relink(link: Path, target: Path):
    """Point ``link`` at ``target`` with a relative symlink, replacing any existing one
    — mirrors the feature dataset into the distilled-label tree without copying the
    (potentially large) feature arrays."""
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = target.resolve().relative_to(link.parent.resolve(), walk_up=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rel)


def load_thresholds(model_name: str) -> dict[str, float]:
    """Read the per-score thresholds picked by autoencoder_test.py."""
    report_path = get_report_dir(RESULTS_DIR / model_name) / AE_TEST_REPORT
    if not report_path.exists():
        raise SystemExit(
            f"no evaluation report at {report_path}. Run autoencoder_test '{model_name}' "
            f"first to pick the thresholds.")
    data = json.loads(report_path.read_text())
    return {n: float(data['thresholds'][n]) for n in SCORE_NAMES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Distill window labels from a trained autoencoder: score the '
                    'mixed-anomaly windows (reconstruction error + spectral / beat-'
                    'interval rhythm indices), label a window anomalous when any score '
                    'crosses the threshold picked by autoencoder_test.py, and write a '
                    'datasets-shaped tree (mixed-features/S*/ with distilled labels.npy '
                    '+ symlinked features) into results/<model>/ that train.py can '
                    'consume via --dataset-dir.')
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to distill from')
    parser.add_argument('--out-subdir', default='distilled-labels',
                        help='Subdirectory of results/<model>/ for the labels (default: distilled-labels)')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    result_dir = RESULTS_DIR / args.model

    thresholds = load_thresholds(args.model)
    # batch_size=1 so every window is scored, no batch remainder dropped — the distilled
    # labels then line up 1:1 with the feature windows.
    trainer = load_autoencoder(args.model, batch_size=1)

    window = trainer.window_size
    if window != BVP_WINDOW:
        raise SystemExit(
            f"model window ({window} samples) does not match the {WINDOW_SECONDS}s feature "
            f"window ({BVP_WINDOW} samples) used to build mixed-features; the autoencoder "
            f"would produce a mismatched number of labels. Align the model's window size.")
    subjects_dir = data_dir / CLEAN_SUBDIR
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = get_sorted_paths(mixed_dir)
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    per_subject: dict[str, np.ndarray] = {}
    print("Labeling windows at thresholds "
          + ", ".join(f"{n}={thresholds[n]:.6f}" for n in SCORE_NAMES) + ":")
    for d in subject_dirs:
        sid = d.name
        signal, cond = conditional_windows(subjects_dir, sid, window, anomalous_dir=mixed_dir)
        n_windows = len(np.load(feature_dir / sid / 'labels.npy').reshape(-1))
        scores = score_windows(trainer.model, signal, cond, window, n_windows)
        flags = predict(scores, thresholds)
        per_subject[sid] = flags
        print(f"  {sid}: {len(flags)} windows, {flags.mean():.1%} flagged")

    # Mirror the feature dataset's structure under out_dir so it can be passed to
    # train.py as a --dataset-dir: only the distilled labels.npy are written; the
    # feature arrays and global stats are symlinked back to the real dataset.
    out_dir = result_dir / args.out_subdir
    out_feature_dir = out_dir / MIXED_FEATURE_SUBDIR
    for sid, flags in per_subject.items():
        labels = flags.astype(np.float32).reshape(-1, 1)
        save_dir = out_feature_dir / sid
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'labels.npy', labels)
        relink(save_dir / 'features.npy', feature_dir / sid / 'features.npy')
    relink(out_feature_dir / FEATURE_STATS_FILE, feature_dir / FEATURE_STATS_FILE)
    np.save(out_dir / 'thresholds.npy',
            np.array([thresholds[n] for n in SCORE_NAMES], dtype=np.float32))
    print(f"Wrote distilled-label dataset for {len(per_subject)} subjects to {out_dir}/")
