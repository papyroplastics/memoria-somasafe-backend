import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.saving import load_trainable_weights
from ml.model_list import MODELS, build_trainer
from ml.models.common import AutoencoderTrainer
from ml.models.cond_lstm_autoencoder import ConditionalAutoencoderTrainer
from ml.data import (SUBJECTS_SUBDIR, ANOMALOUS_SUBDIR, FEATURE_SUBDIR,
                     FEATURE_STATS_FILE, stacked_signal, norm_stats, normalize)
from ml.metrics import best_threshold, classification_report

SEED = 1234


def relink(link: Path, target: Path):
    """Point ``link`` at ``target`` with a relative symlink, replacing any existing
    one. Used to mirror the feature dataset into the distilled-label tree without
    copying the (potentially large) feature arrays."""
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = target.resolve().relative_to(link.parent.resolve(), walk_up=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(rel)


def window_errors(model, signal: np.ndarray, window: int, n_windows: int) -> np.ndarray:
    """Reconstruction error for the first ``n_windows`` non-overlapping windows of a
    normalized ``[BVP, ACC]`` signal — the autoencoder's per-window anomaly score.
    Built with a batch-size-1 model so each window scores independently."""
    errors = np.empty(n_windows, dtype=np.float32)
    for w in range(n_windows):
        s = w * window
        win = signal[s:s + window]
        out = model.eval(win[None].astype(np.float32))
        errors[w] = float(out['error'][0])
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Distill window labels from a trained autoencoder: score the '
                    'synthetic-anomaly windows by reconstruction error, pick the '
                    'accuracy-maximizing threshold against the ground-truth labels, '
                    'and write a datasets-shaped tree (anomalous-features/S*/ with '
                    'distilled labels.npy + symlinked features) into results/<model>/ '
                    'that train.py can consume via --dataset-dir.')
    parser.add_argument('model', choices=sorted(MODELS), help='Trained autoencoder to distill from')
    parser.add_argument('--out-subdir', default='distilled-labels',
                        help='Subdirectory of results/<model>/ for the labels (default: distilled-labels)')
    args = parser.parse_args()

    data_dir = Path('datasets')
    result_dir = Path('results') / args.model

    # Batch size 1 so we can score windows one at a time regardless of how the
    # model was trained; trainable weights are batch-size independent.
    trainer = build_trainer(args.model, data_dir, SEED, batch_size=1)
    if not isinstance(trainer, AutoencoderTrainer) or isinstance(trainer, ConditionalAutoencoderTrainer):
        raise SystemExit(
            f"'{args.model}' is not a non-conditional autoencoder; distillation needs "
            f"one (lstm-ae, gru-ae, cnn-ae).")

    weights_path = result_dir / 'trainable.tflite'
    if not weights_path.exists():
        raise SystemExit(f"trained model not found at {weights_path}. Train '{args.model}' first.")
    trainer.model.restore(tf.constant(load_trainable_weights(weights_path)))

    window = trainer.window_size
    subjects_dir = data_dir / SUBJECTS_SUBDIR
    anomalous_dir = data_dir / ANOMALOUS_SUBDIR
    feature_dir = data_dir / FEATURE_SUBDIR

    subject_dirs = sorted(anomalous_dir.glob('S*'))
    if not subject_dirs:
        raise SystemExit(f"{anomalous_dir} is empty. Run get_dataset.py first.")

    mean, std = norm_stats(subjects_dir)

    per_subject: dict[str, np.ndarray] = {}
    all_err: list[np.ndarray] = []
    all_lbl: list[np.ndarray] = []
    print("Scoring windows by reconstruction error:")
    for d in subject_dirs:
        sid = d.name
        signal = normalize(stacked_signal(subjects_dir, sid, anomalous_dir=anomalous_dir),
                           mean, std)
        truth = np.load(feature_dir / sid / 'labels.npy').reshape(-1)  # per-window ground truth
        errs = window_errors(trainer.model, signal, window, len(truth))
        per_subject[sid] = errs
        all_err.append(errs)
        all_lbl.append(truth)
        print(f"  {sid}: {len(truth)} windows")

    errors = np.concatenate(all_err)
    truth = np.concatenate(all_lbl)
    thr, acc = best_threshold(errors, truth)

    pred = errors > thr
    report = classification_report(pred, truth)
    print(f"\nthreshold={thr:.6f} accuracy={acc:.4f} "
          f"precision={report['precision']:.4f} recall={report['recall']:.4f}")
    print(f"ground-truth anomaly rate={truth.mean():.1%}  distilled rate={pred.mean():.1%}")

    # Mirror the feature dataset's structure under out_dir so it can be passed to
    # train.py as a --dataset-dir: only the distilled labels.npy are written; the
    # feature arrays and global stats are symlinked back to the real dataset.
    out_dir = result_dir / args.out_subdir
    out_feature_dir = out_dir / FEATURE_SUBDIR
    for sid, errs in per_subject.items():
        labels = (errs > thr).astype(np.float32).reshape(-1, 1)
        save_dir = out_feature_dir / sid
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'labels.npy', labels)
        relink(save_dir / 'features.npy', feature_dir / sid / 'features.npy')
    relink(out_feature_dir / FEATURE_STATS_FILE, feature_dir / FEATURE_STATS_FILE)
    np.save(out_dir / 'threshold.npy', np.array([thr], dtype=np.float32))
    print(f"Wrote distilled-label dataset for {len(per_subject)} subjects to {out_dir}/")
