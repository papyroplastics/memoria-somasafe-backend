import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.saving import load_trainable_weights
from ml.model_list import MODELS, build_trainer
from ml.models.common import AutoencoderTrainer
from ml.models.cond_lstm_autoencoder import ConditionalAutoencoderTrainer
from scripts.get_dataset import NORMALIZED_ANOMALOUS_SUBDIR

SEED = 1234


def window_errors(model, bvp: np.ndarray, acc: np.ndarray,
                  window: int, n_windows: int) -> np.ndarray:
    """Reconstruction error for the first ``n_windows`` non-overlapping
    ``[BVP, ACC]`` windows — the autoencoder's per-window anomaly score. Built
    with a batch-size-1 model so each window scores independently."""
    errors = np.empty(n_windows, dtype=np.float32)
    for w in range(n_windows):
        s = w * window
        win = np.stack([bvp[s:s + window], acc[s:s + window]], axis=-1)
        out = model.eval(win[None].astype(np.float32))
        errors[w] = float(out['error'][0])
    return errors


def best_threshold(errors: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Reconstruction-error threshold (predict anomalous when error > threshold)
    that maximizes accuracy against ``labels``. Single sorted sweep."""
    order = np.argsort(errors, kind='stable')
    e = errors[order]
    y = labels[order].astype(bool)
    n = len(y)

    # Split i predicts windows [i, n) anomalous; accuracy = correct negatives in
    # [0, i) + correct positives in [i, n).
    cumneg = np.concatenate([[0], np.cumsum(~y)])          # negatives in [0, i)
    pos_suffix = int(y.sum()) - np.concatenate([[0], np.cumsum(y)])  # positives in [i, n)
    acc = (cumneg + pos_suffix) / n

    i = int(np.argmax(acc))
    if i == 0:
        thr = float(e[0]) - 1.0
    elif i == n:
        thr = float(e[-1]) + 1.0
    else:
        thr = float((e[i - 1] + e[i]) / 2.0)
    return thr, float(acc[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Distill window labels from a trained autoencoder: score the '
                    'synthetic-anomaly windows by reconstruction error, pick the '
                    'accuracy-maximizing threshold against the ground-truth labels, '
                    'and write feature-mlp-shaped labels into results/<model>/.')
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
    norm_anom_dir = data_dir / NORMALIZED_ANOMALOUS_SUBDIR
    feature_dir = data_dir / 'feature-anomaly'

    subject_dirs = sorted(norm_anom_dir.glob('S*'))
    if not subject_dirs:
        raise SystemExit(f"{norm_anom_dir} is empty. Run get_dataset.py first.")

    per_subject: dict[str, np.ndarray] = {}
    all_err: list[np.ndarray] = []
    all_lbl: list[np.ndarray] = []
    print("Scoring windows by reconstruction error:")
    for d in subject_dirs:
        sid = d.name
        bvp = np.load(d / 'bvp.npy')
        acc = np.load(d / 'acc.npy')
        truth = np.load(feature_dir / sid / 'labels.npy').reshape(-1)  # per-window ground truth
        errs = window_errors(trainer.model, bvp, acc, window, len(truth))
        per_subject[sid] = errs
        all_err.append(errs)
        all_lbl.append(truth)
        print(f"  {sid}: {len(truth)} windows")

    errors = np.concatenate(all_err)
    truth = np.concatenate(all_lbl)
    thr, acc = best_threshold(errors, truth)

    pred = errors > thr
    tp = int(np.sum(pred & (truth > 0.5)))
    fp = int(np.sum(pred & (truth < 0.5)))
    fn = int(np.sum(~pred & (truth > 0.5)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\nthreshold={thr:.6f} accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f}")
    print(f"ground-truth anomaly rate={truth.mean():.1%}  distilled rate={pred.mean():.1%}")

    out_dir = result_dir / args.out_subdir
    for sid, errs in per_subject.items():
        labels = (errs > thr).astype(np.float32).reshape(-1, 1)
        save_dir = out_dir / sid
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / 'labels.npy', labels)
    np.save(out_dir / 'threshold.npy', np.array([thr], dtype=np.float32))
    print(f"Wrote distilled labels for {len(per_subject)} subjects to {out_dir}/")
