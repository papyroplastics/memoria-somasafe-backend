"""Exports one subject's windowed dataset to the ``somasafe.capture.CaptureDataset``
protobuf that the Android app imports and the firmware harness streams, skipping the slow
UART -> ESP -> BLE streaming path, with each window stored exactly as the app's capture
schema stores an ESP sample and faked recording metadata assigned before anything is
dropped so the missing-data flags leave real capture-like sequence holes."""

import argparse
from pathlib import Path

import numpy as np

from common.config import DATASETS_DIR, EXPORTS_DIR
from ml.dataset_list import DATASETS
from ml.preprocessing import BVP_WINDOW, ACC_WINDOW, WINDOW_SECONDS
from ml.sources.dalia import CLEAN, MIXED
from shared.gen.code import dataset_pb2 as pb

FORMAT_VERSION = 1

def window_raw(signal: np.ndarray, window: int, count: int) -> np.ndarray:
    frames = signal[: count * window].reshape(count, window)
    return frames.astype(np.float32)


def present_mask(count: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Returns a boolean mask with round(fraction * count) entries True at random positions."""
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
    variant = CLEAN if clean else MIXED
    source = DATASETS['ppg-dalia'].build(datasets_dir)

    bvp      = source.signal(sid, variant).astype(np.float32)
    acc      = source.acc_signal(sid).astype(np.float32)
    features = source.raw_features(sid, variant).astype(np.float32)
    labels   = (source.window_labels(sid) if variant == MIXED
                else np.zeros(len(features), dtype=np.float32))

    count = min(len(bvp) // BVP_WINDOW, len(acc) // ACC_WINDOW, len(features), len(labels))
    if count == 0:
        raise ValueError(f"{sid}: no complete windows to export")

    ppg_win  = window_raw(bvp, BVP_WINDOW, count)
    acc_win  = window_raw(acc, ACC_WINDOW, count)
    feat_win = features[:count].astype(np.float32)
    score    = labels[:count].astype(np.int8).reshape(count, 1)

    rng = np.random.default_rng()
    window_ms = np.uint64(WINDOW_SECONDS * 1000)
    base_ms = np.uint64(rng.integers(0, 2**31 - count * int(window_ms)))
    seq      = np.arange(count, dtype=np.uint32)
    dev_start = (base_ms + seq.astype(np.uint64) * window_ms).astype(np.uint32)
    dev_end   = (base_ms + (seq.astype(np.uint64) + np.uint64(1)) * window_ms).astype(np.uint32)

    if missing_samples is None and missing_features is None:
        data_present = np.ones(count, dtype=bool)
        feat_present = np.ones(count, dtype=bool)
    elif missing_features is None:
        data_present = present_mask(count, 1.0 - missing_samples, rng)
        feat_present = data_present
    elif missing_samples is None:
        data_present = np.ones(count, dtype=bool)
        feat_present = present_mask(count, 1.0 - missing_features, rng)
    else:
        data_present = present_mask(count, 1.0 - missing_samples, rng)
        feat_present = present_mask(count, 1.0 - missing_features, rng)

    dataset = pb.CaptureDataset(format_version=FORMAT_VERSION, subject=subject)
    for i in range(count):
        if not (data_present[i] or feat_present[i]):
            continue
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument('subject', type=int, nargs='?', default=1, help="Subject id to export")
    parser.add_argument('--datasets-dir', type=Path, default=DATASETS_DIR,
                        help="Datasets directory")
    parser.add_argument('-o', '--output', type=Path, default=None, help="Output file")
    parser.add_argument('--clean', action='store_true',
                        help="Export the clean signal dataset instead of mixed")
    parser.add_argument('--missing-samples', type=unit_fraction, default=None, metavar='F',
                        help="Drop a random fraction F of windows' signal data")
    parser.add_argument('--missing-features', type=unit_fraction, default=None, metavar='F',
                        help="Drop a random fraction F of windows' ML result")
    args = parser.parse_args()

    out = args.output or EXPORTS_DIR / f'S{args.subject}.ssds'
    export_subject(args.subject, args.datasets_dir, out,
                   clean=args.clean,
                   missing_samples=args.missing_samples,
                   missing_features=args.missing_features)
