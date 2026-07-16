"""Export one subject's windowed dataset to the ``somasafe.capture.CaptureDataset``
protobuf (``shared/dataset.proto``, run ``make proto`` to regenerate the bindings) that
the Android app imports, skipping the slow UART -> ESP -> BLE streaming path. Each
window is stored exactly as the app's capture schema stores an ESP sample — raw float32
PPG/ACC, the raw feature vector, the label in the int8 score field, and faked recording
metadata (sequence numbers plus contiguous 8 s device timestamps from a random boot
offset) assigned to the full window grid *before* anything is dropped, so
``--missing-samples F`` / ``--missing-features F`` (independently drop a random fraction
F of the windows' signal data / ML result) leave real capture-like sequence holes.
``--clean`` exports the clean signal dataset instead of mixed; its features/labels come
from the ``clean-features`` dataset (all windows normal, score 0), precomputed by
scripts/get_dataset.py so the phone can skip the slow on-device extraction, so
``--missing-features`` works with ``--clean`` too.
"""

import argparse
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR
from ml.preprocessing import (
    MIXED_SUBDIR, CLEAN_SUBDIR, MIXED_FEATURE_SUBDIR, CLEAN_FEATURE_SUBDIR,
    BVP_WINDOW, ACC_WINDOW, WINDOW_SECONDS,
)
from ..common import dataset_pb2 as pb

FORMAT_VERSION = 1

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
                   clean: bool = False,
                   missing_samples: float | None = None,
                   missing_features: float | None = None):
    sid = f'S{subject}'
    signal_subdir = CLEAN_SUBDIR if clean else MIXED_SUBDIR
    feature_subdir = CLEAN_FEATURE_SUBDIR if clean else MIXED_FEATURE_SUBDIR
    bvp_path   = datasets_dir / signal_subdir   / sid / 'bvp.npy'
    acc_path   = datasets_dir / CLEAN_SUBDIR    / sid / 'acc.npy'
    feat_path  = datasets_dir / feature_subdir  / sid / 'features.npy'
    label_path = datasets_dir / feature_subdir  / sid / 'labels.npy'
    for path in [bvp_path, acc_path, feat_path, label_path]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/get_dataset.py first.")

    bvp      = np.load(bvp_path).astype(np.float32)
    acc      = np.load(acc_path).astype(np.float32)
    features = np.load(feat_path).astype(np.float32)                 # (N, 17) raw
    labels   = np.load(label_path).astype(np.float32).reshape(-1)

    # Align to the windows every source agrees on (same indexing as build_feature_dataset).
    count = min(len(bvp) // BVP_WINDOW, len(acc) // ACC_WINDOW, len(features), len(labels))
    if count == 0:
        raise ValueError(f"{sid}: no complete windows to export")

    ppg_win  = window_raw(bvp, BVP_WINDOW, count)             # (count, 512) f32
    acc_win  = window_raw(acc, ACC_WINDOW, count)             # (count, 256) f32
    feat_win = features[:count].astype(np.float32)           # (count, 17) f32
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

    # Decide which windows keep their signal and which keep their ML result
    # (--missing-* give the fraction to drop, so the kept fraction is 1 - F).
    if missing_samples is None and missing_features is None:
        data_present = np.ones(count, dtype=bool)
        feat_present = np.ones(count, dtype=bool)
    elif missing_features is None:                 # only --missing-samples
        data_present = present_mask(count, 1.0 - missing_samples, rng)
        feat_present = data_present                 # features ride along with the window
    elif missing_samples is None:                  # only --missing-features
        data_present = np.ones(count, dtype=bool)
        feat_present = present_mask(count, 1.0 - missing_features, rng)
    else:                                          # both, independent
        data_present = present_mask(count, 1.0 - missing_samples, rng)
        feat_present = present_mask(count, 1.0 - missing_features, rng)

    dataset = pb.CaptureDataset(format_version=FORMAT_VERSION, subject=subject)
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
    parser.add_argument('--clean', action='store_true',
                        help="Export the clean signal dataset (no injected anomalies) instead of "
                             "mixed; features/labels come from clean-features (all windows normal)")
    parser.add_argument('--missing-samples', type=unit_fraction, default=None, metavar='F',
                        help="Drop a random fraction F of windows' signal data")
    parser.add_argument('--missing-features', type=unit_fraction, default=None, metavar='F',
                        help="Drop a random fraction F of windows' ML result (features + score)")
    args = parser.parse_args()

    out = args.output or Path(f'S{args.subject}.ssds')
    export_subject(args.subject, args.datasets_dir, out,
                   clean=args.clean,
                   missing_samples=args.missing_samples,
                   missing_features=args.missing_features)
