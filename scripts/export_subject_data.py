"""Export one subject's windowed dataset to the protobuf the Android app imports.

The phone normally builds a dataset by streaming a subject through UART -> ESP ->
BLE, which is slow. This produces the same per-window rows directly so they can be
imported on-device for training tests. Each window is written exactly as the app's
capture schema stores an ESP sample: raw float32 PPG/ACC, the raw (un-normalized)
float32 feature vector the device echoes, and the label in the int8 score field.

The wire format is the ``somasafe.capture.CaptureDataset`` message defined in
``shared/dataset.proto`` (run ``make proto`` to regenerate ``scripts/common/dataset_pb2.py``).

With ``--include-context`` each window also carries its raw (un-normalized) 2-d
activity context (context.npy); otherwise the context field is empty and the phone
computes context itself.

Every window carries the recording-intrinsic metadata an ESP sample has: its
sequence number and on-device start/end timestamps. The timestamps are faked on an
exact 8-second grid from a random boot offset (the ESP clock starts at 0 on boot), so
imported windows are contiguous in device time and the phone's context pass treats
them exactly like a live capture. The receive-time is the phone's to stamp, so it is
synthesized at import (not stored here).

``--missing-samples`` and ``--missing-features`` simulate packet loss. Sequence numbers
and timestamps are assigned to the full window grid *before* anything is dropped, so a
removed window leaves a real hole in the sequence numbers, exactly like a lost capture:

  * ``--missing-samples F``  keep a random fraction F of windows' signal data; the rest
    lose their PPG/ACC (and, with no ``--missing-features``, their features too, so the
    window vanishes entirely). Device timestamps go with the signal, mirroring how a
    result-only row in the app has no device time.
  * ``--missing-features F`` keep a random fraction F of windows' ML result (features +
    score); the rest keep their signal but arrive without features for the phone to
    recompute.

Passing both draws the two sets independently, so a window may end up with signal but
no features, features but no signal, or neither (in which case it is omitted).
"""

import argparse
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR
from ml.data import (
    MIXED_SUBDIR, CLEAN_SUBDIR, MIXED_FEATURE_SUBDIR, CONTEXT_FILE,
    BVP_WINDOW, ACC_WINDOW, WINDOW_SECONDS, load_static_norm_params,
)
from .common import dataset_pb2 as pb

FORMAT_VERSION = 4


def window_raw(signal: np.ndarray, window: int, count: int) -> np.ndarray:
    frames = signal[: count * window].reshape(count, window)
    return frames.astype(np.float32)


def present_mask(count: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask with round(fraction * count) entries True at random positions."""
    keep = min(count, max(0, round(fraction * count)))
    mask = np.zeros(count, dtype=bool)
    if keep:
        mask[rng.choice(count, size=keep, replace=False)] = True
    return mask


def export_subject(subject: int, datasets_dir: Path, out_path: Path,
                   include_context: bool = False,
                   missing_samples: float | None = None,
                   missing_features: float | None = None):
    sid = f'S{subject}'
    bvp_path   = datasets_dir / MIXED_SUBDIR         / sid / 'bvp.npy'
    acc_path   = datasets_dir / CLEAN_SUBDIR      / sid / 'acc.npy'
    feat_path  = datasets_dir / MIXED_FEATURE_SUBDIR / sid / 'features.npy'
    label_path = datasets_dir / MIXED_FEATURE_SUBDIR / sid / 'labels.npy'
    ctx_path   = datasets_dir / CLEAN_SUBDIR       / sid / CONTEXT_FILE
    static_path = datasets_dir / CLEAN_SUBDIR      / sid / 'static.npy'
    required = [bvp_path, acc_path, feat_path, label_path, static_path]
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

    # Recover the subject's raw 6-d demographics by de-normalizing the stored static
    # vector: static.npy is z-scored, so raw = norm * std + mean. Ships raw so the phone
    # re-normalizes with the same static_norm_params it pulls from /model/norm.
    stat_mean, stat_std = load_static_norm_params(datasets_dir / CLEAN_SUBDIR)
    static_norm = np.load(static_path).astype(np.float32)     # (6,) z-scored
    static_raw = (static_norm * stat_std + stat_mean).astype(np.float32)

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

    # Fake ESP metadata over the full grid, before any window is dropped: sequence
    # numbers 0..count-1 and contiguous 8 s device-time windows from a random boot
    # offset (kept well within u32 so the device clock fits).
    rng = np.random.default_rng()
    window_ms = np.uint64(WINDOW_SECONDS * 1000)
    base_ms = np.uint64(rng.integers(0, 2**31 - count * int(window_ms)))
    seq      = np.arange(count, dtype=np.uint32)
    dev_start = (base_ms + seq.astype(np.uint64) * window_ms).astype(np.uint32)
    dev_end   = (base_ms + (seq.astype(np.uint64) + np.uint64(1)) * window_ms).astype(np.uint32)

    # Decide which windows keep their signal and which keep their ML result.
    if missing_samples is None and missing_features is None:
        data_present = np.ones(count, dtype=bool)
        feat_present = np.ones(count, dtype=bool)
    elif missing_features is None:                 # only --missing-samples
        data_present = present_mask(count, missing_samples, rng)
        feat_present = data_present                 # features ride along with the window
    elif missing_samples is None:                  # only --missing-features
        data_present = np.ones(count, dtype=bool)
        feat_present = present_mask(count, missing_features, rng)
    else:                                          # both, independent
        data_present = present_mask(count, missing_samples, rng)
        feat_present = present_mask(count, missing_features, rng)

    dataset = pb.CaptureDataset(format_version=FORMAT_VERSION, subject=subject,
                                static=static_raw.tobytes())
    for i in range(count):
        if not (data_present[i] or feat_present[i]):
            continue                                # neither half survived: real seq gap
        w = dataset.windows.add()
        w.sequence_n = int(seq[i])
        if data_present[i]:
            w.device_start_ms = int(dev_start[i])
            w.device_end_ms = int(dev_end[i])
            w.ppg = ppg_win[i].tobytes()
            w.acc = acc_win[i].tobytes()
            if ctx_win is not None:
                w.context = ctx_win[i].tobytes()
        if feat_present[i]:
            w.features = feat_win[i].tobytes()
            w.score = score[i].tobytes()

    out_path.write_bytes(dataset.SerializeToString())

    written = len(dataset.windows)
    data_only = int(np.sum(data_present & ~feat_present))
    feat_only = int(np.sum(feat_present & ~data_present))
    anomalous = int(score[feat_present].sum())
    size = out_path.stat().st_size
    print(f"{sid}: {written}/{count} windows -> {out_path} ({size} bytes); "
          f"{data_only} signal-only, {feat_only} result-only, {anomalous} anomalous")


def unit_fraction(value: str) -> float:
    f = float(value)
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError("must be a fraction between 0 and 1")
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', type=int, help="Subject id to export (e.g. 1)")
    parser.add_argument('--datasets-dir', type=Path, default=DATASETS_DIR,
                        help=f"Datasets directory (default: {DATASETS_DIR})")
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help="Output file (default: S<id>.ssds)")
    parser.add_argument('--include-context', action='store_true',
                        help="Embed each window's raw activity context (context.npy)")
    parser.add_argument('--missing-samples', type=unit_fraction, default=None, metavar='F',
                        help="Keep a random fraction F of windows' signal data; drop the rest")
    parser.add_argument('--missing-features', type=unit_fraction, default=None, metavar='F',
                        help="Keep a random fraction F of windows' ML result (features + score)")
    args = parser.parse_args()

    out = args.output or Path(f'S{args.subject}.ssds')
    export_subject(args.subject, args.datasets_dir, out,
                   include_context=args.include_context,
                   missing_samples=args.missing_samples,
                   missing_features=args.missing_features)
