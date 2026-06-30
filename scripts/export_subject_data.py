"""Export one subject's windowed dataset to a flat binary the Android app imports.

The phone normally builds a dataset by streaming a subject through UART -> ESP ->
BLE, which is slow. This produces the same per-window rows directly so they can be
imported on-device for training tests. Each window is written exactly as the app's
capture schema stores an ESP sample: raw float32 PPG/ACC, the raw (un-normalized)
float32 feature vector the device echoes, and the label in the int8 score field.

With ``--include-context`` each window also carries its raw (un-normalized) 2-d
activity context (context.npy); otherwise the context field is empty (ctxLen 0) and
the phone computes context itself.

Each window also carries the recording-intrinsic metadata an ESP sample has: its
sequence number and on-device start/end timestamps. The timestamps are faked on an
exact 8-second grid from a random boot offset (the ESP clock starts at 0 on boot), so
imported windows are contiguous in device time and the phone's context pass treats
them exactly like a live capture. The receive-time is the phone's to stamp, so it is
synthesized at import (not stored here).
"""

import argparse
import struct
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR
from ml.data import (
    MIXED_SUBDIR, CLEAN_SUBDIR, MIXED_FEATURE_SUBDIR, CONTEXT_FILE,
    BVP_WINDOW, ACC_WINDOW, WINDOW_SECONDS,
)

MAGIC = b'SSDS'
VERSION = 2


def window_raw(signal: np.ndarray, window: int, count: int) -> np.ndarray:
    frames = signal[: count * window].reshape(count, window)
    return frames.astype(np.float32)


def export_subject(subject: int, datasets_dir: Path, out_path: Path,
                   include_context: bool = False):
    sid = f'S{subject}'
    bvp_path   = datasets_dir / MIXED_SUBDIR         / sid / 'bvp.npy'
    acc_path   = datasets_dir / CLEAN_SUBDIR      / sid / 'acc.npy'
    feat_path  = datasets_dir / MIXED_FEATURE_SUBDIR / sid / 'features.npy'
    label_path = datasets_dir / MIXED_FEATURE_SUBDIR / sid / 'labels.npy'
    ctx_path   = datasets_dir / CLEAN_SUBDIR       / sid / CONTEXT_FILE
    required = [bvp_path, acc_path, feat_path, label_path]
    if include_context:
        required.append(ctx_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/get_dataset.py first.")

    bvp      = np.load(bvp_path).astype(np.float32)
    acc      = np.load(acc_path).astype(np.float32)
    features = np.load(feat_path).astype(np.float32)          # (N, 17) raw
    labels   = np.load(label_path).astype(np.float32).reshape(-1)
    context  = np.load(ctx_path).astype(np.float32) if include_context else None  # (N, 2) raw

    # Align to the windows every source agrees on (same indexing as build_feature_dataset).
    lengths = [len(features), len(labels), len(bvp) // BVP_WINDOW, len(acc) // ACC_WINDOW]
    if context is not None:
        lengths.append(len(context))
    count = min(lengths)
    if count == 0:
        raise ValueError(f"{sid}: no complete windows to export")

    ppg_win  = window_raw(bvp, BVP_WINDOW, count)             # (count, 512) f32
    acc_win  = window_raw(acc, ACC_WINDOW, count)             # (count, 256) f32
    feat_win = features[:count].astype(np.float32)            # (count, 17) f32
    ctx_win  = context[:count].astype(np.float32) if context is not None else None  # (count, 2) f32
    score    = labels[:count].astype(np.int8).reshape(count, 1)  # (count, 1) int8

    # Fake ESP metadata: sequence numbers 0..count-1 and contiguous 8 s device-time
    # windows from a random boot offset (kept well within u32 so the device clock fits).
    window_ms = np.uint64(WINDOW_SECONDS * 1000)
    base_ms = np.uint64(np.random.default_rng().integers(0, 2**31 - count * int(window_ms)))
    seq      = np.arange(count, dtype=np.uint32)
    dev_start = (base_ms + seq.astype(np.uint64) * window_ms).astype(np.uint32)
    dev_end   = (base_ms + (seq.astype(np.uint64) + np.uint64(1)) * window_ms).astype(np.uint32)

    feat_len  = feat_win.shape[1] * 4
    ctx_len   = ctx_win.shape[1] * 4 if ctx_win is not None else 0
    score_len = score.shape[1]
    ppg_bytes = BVP_WINDOW * 4
    acc_bytes = ACC_WINDOW * 4

    header = MAGIC + struct.pack(
        '<BHHHHHHI', VERSION, subject, ppg_bytes, acc_bytes, feat_len, ctx_len, score_len, count,
    )
    with open(out_path, 'wb') as f:
        f.write(header)
        for i in range(count):
            f.write(struct.pack('<III', int(seq[i]), int(dev_start[i]), int(dev_end[i])))
            f.write(ppg_win[i].tobytes())
            f.write(acc_win[i].tobytes())
            f.write(feat_win[i].tobytes())
            if ctx_win is not None:
                f.write(ctx_win[i].tobytes())
            f.write(score[i].tobytes())

    anomalous = int(score.sum())
    size = out_path.stat().st_size
    print(f"{sid}: {count} windows ({anomalous} anomalous) -> {out_path} ({size} bytes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', type=int, help="Subject id to export (e.g. 1)")
    parser.add_argument('--datasets-dir', type=Path, default=DATASETS_DIR,
                        help=f"Datasets directory (default: {DATASETS_DIR})")
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help="Output file (default: S<id>.ssds)")
    parser.add_argument('--include-context', action='store_true',
                        help="Embed each window's raw activity context (context.npy)")
    args = parser.parse_args()

    out = args.output or Path(f'S{args.subject}.ssds')
    export_subject(args.subject, args.datasets_dir, out, include_context=args.include_context)
