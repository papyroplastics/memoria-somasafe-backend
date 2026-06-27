import argparse
import json
from pathlib import Path

import numpy as np

from common.config import RESULTS_DIR, DATASETS_DIR
from ml.data import (SUBJECTS_SUBDIR, MIXED_SUBDIR, MIXED_FEATURE_SUBDIR,
                     FEATURE_STATS_FILE, stacked_signal, norm_stats, normalize)
from ml.model_list import MODELS
from .test_autoencoder import load_autoencoder, window_errors
from .common.post_train import get_report_dir, AE_TEST_REPORT


def relink(link: Path, target: Path):
    """Point ``link`` at ``target`` with a relative symlink, replacing any existing
    one. Used to mirror the feature dataset into the distilled-label tree without
    copying the (potentially large) feature arrays."""
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = target.resolve().relative_to(link.parent.resolve(), walk_up=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rel)


def load_threshold(model_name: str) -> float:
    """Read the reconstruction-error threshold picked by test_autoencoder.py."""
    report_path = get_report_dir(RESULTS_DIR / model_name) / AE_TEST_REPORT
    if not report_path.exists():
        raise SystemExit(
            f"no evaluation report at {report_path}. Run test_autoencoder '{model_name}' "
            f"first to pick the threshold.")
    return float(json.loads(report_path.read_text())['threshold'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Distill window labels from a trained autoencoder: score the '
                    'synthetic-anomaly windows by reconstruction error, label them '
                    'with the threshold chosen by test_autoencoder.py, and write a '
                    'datasets-shaped tree (mixed-features/S*/ with distilled '
                    'labels.npy + symlinked features) into results/<model>/ that '
                    'train.py can consume via --dataset-dir.')
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to distill from')
    parser.add_argument('--out-subdir', default='distilled-labels',
                        help='Subdirectory of results/<model>/ for the labels (default: distilled-labels)')
    args = parser.parse_args()

    data_dir = DATASETS_DIR
    result_dir = RESULTS_DIR / args.model

    thr = load_threshold(args.model)
    trainer = load_autoencoder(args.model, data_dir)

    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR
    mixed_dir = data_dir / MIXED_SUBDIR
    feature_dir = data_dir / MIXED_FEATURE_SUBDIR

    subject_dirs = sorted(mixed_dir.glob('S*'))
    if not subject_dirs:
        raise SystemExit(f"{mixed_dir} is empty. Run get_dataset.py first.")

    mean, std = norm_stats(subjects_dir)

    per_subject: dict[str, np.ndarray] = {}
    print(f"Labeling windows at threshold={thr:.6f}:")
    for d in subject_dirs:
        sid = d.name
        signal = normalize(stacked_signal(subjects_dir, sid, anomalous_dir=mixed_dir),
                           mean, std)
        n_windows = len(np.load(feature_dir / sid / 'labels.npy').reshape(-1))
        errs = window_errors(trainer.model, signal, window, n_windows)
        per_subject[sid] = errs
        print(f"  {sid}: {n_windows} windows")

    # Mirror the feature dataset's structure under out_dir so it can be passed to
    # train.py as a --dataset-dir: only the distilled labels.npy are written; the
    # feature arrays and global stats are symlinked back to the real dataset.
    out_dir = result_dir / args.out_subdir
    out_feature_dir = out_dir / MIXED_FEATURE_SUBDIR
    for sid, errs in per_subject.items():
        labels = (errs > thr).astype(np.float32).reshape(-1, 1)
        save_dir = out_feature_dir / sid
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'labels.npy', labels)
        relink(save_dir / 'features.npy', feature_dir / sid / 'features.npy')
    relink(out_feature_dir / FEATURE_STATS_FILE, feature_dir / FEATURE_STATS_FILE)
    np.save(out_dir / 'threshold.npy', np.array([thr], dtype=np.float32))
    print(f"Wrote distilled-label dataset for {len(per_subject)} subjects to {out_dir}/")
