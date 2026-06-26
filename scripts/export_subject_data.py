"""Export one subject's windowed dataset to a flat binary the Android app imports.

The phone normally builds a dataset by streaming a subject through UART -> ESP ->
BLE, which is slow. This produces the same per-window rows directly so they can be
imported on-device for training tests. Each window is written exactly as the app's
capture schema stores an ESP sample: raw float32 PPG/ACC, the int8 feature vector
the device echoes (normalized features quantized with the model's input params),
and the label in the int8 score field.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf

from common.config import DATASETS_DIR
from ml.data import (
    ANOMALOUS_SUBDIR, SUBJECTS_SUBDIR, FEATURE_SUBDIR,
    BVP_WINDOW, ACC_WINDOW,
)

MAGIC = b'SSDS'
VERSION = 1


def quantize_features(features: np.ndarray, model_path: Path) -> np.ndarray:
    """Quantize normalized float32 features to the int8 input the device echoes,
    using the quantized model's input (scale, zero_point)."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    scale, zero_point = interp.get_input_details()[0]['quantization']
    if scale == 0:
        raise ValueError(f"{model_path} has no input quantization (scale=0)")
    q = np.round(features / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def window_raw(signal: np.ndarray, window: int, count: int) -> np.ndarray:
    """First `count` non-overlapping `window`-sample frames as float32."""
    frames = signal[: count * window].reshape(count, window)
    return frames.astype(np.float32)


def export_subject(subject: int, model_path: Path, datasets_dir: Path, out_path: Path):
    sid = f'S{subject}'
    bvp_path   = datasets_dir / ANOMALOUS_SUBDIR / sid / 'bvp.npy'
    acc_path   = datasets_dir / SUBJECTS_SUBDIR  / sid / 'acc.npy'
    feat_path  = datasets_dir / FEATURE_SUBDIR   / sid / 'features.npy'
    label_path = datasets_dir / FEATURE_SUBDIR   / sid / 'labels.npy'
    for path in (bvp_path, acc_path, feat_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/get_dataset.py first.")

    bvp      = np.load(bvp_path).astype(np.float32)
    acc      = np.load(acc_path).astype(np.float32)
    features = np.load(feat_path).astype(np.float32)          # (N, 17) normalized
    labels   = np.load(label_path).astype(np.float32).reshape(-1)

    # Align to the windows every source agrees on (same indexing as build_feature_dataset).
    count = min(len(features), len(labels), len(bvp) // BVP_WINDOW, len(acc) // ACC_WINDOW)
    if count == 0:
        raise ValueError(f"{sid}: no complete windows to export")

    ppg_win  = window_raw(bvp, BVP_WINDOW, count)             # (count, 512) f32
    acc_win  = window_raw(acc, ACC_WINDOW, count)             # (count, 256) f32
    feat_q   = quantize_features(features[:count], model_path)  # (count, 17) int8
    score    = labels[:count].astype(np.int8).reshape(count, 1)  # (count, 1) int8

    feat_len  = feat_q.shape[1]
    score_len = score.shape[1]
    ppg_bytes = BVP_WINDOW * 4
    acc_bytes = ACC_WINDOW * 4

    header = MAGIC + struct.pack(
        '<BHHHHHI', VERSION, subject, ppg_bytes, acc_bytes, feat_len, score_len, count,
    )
    with open(out_path, 'wb') as f:
        f.write(header)
        for i in range(count):
            f.write(ppg_win[i].tobytes())
            f.write(acc_win[i].tobytes())
            f.write(feat_q[i].tobytes())
            f.write(score[i].tobytes())

    anomalous = int(score.sum())
    size = out_path.stat().st_size
    print(f"{sid}: {count} windows ({anomalous} anomalous) -> {out_path} ({size} bytes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int, help="Subject id to export (e.g. 1)")
    parser.add_argument('model', type=Path, help="Path to the int8 quantized .tflite")
    parser.add_argument('--datasets-dir', type=Path, default=DATASETS_DIR,
                        help=f"Datasets directory (default: {DATASETS_DIR})")
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help="Output file (default: S<id>.ssds)")
    args = parser.parse_args()

    out = args.output or Path(f'S{args.subject}.ssds')
    export_subject(args.subject, args.model, args.datasets_dir, out)
