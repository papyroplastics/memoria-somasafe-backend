from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common.config import SEED

from .common import DataSource
from ..spectral import descriptors as window_descriptors
from ..preprocessing import (
    ACC_WINDOW, ACTIVITY_FILE, ANOMALOUS_SUBDIR, ANOMALY_KINDS, BVP_WINDOW,
    CLEAN_SUBDIR, N_FEATURES, DatasetUnavailibleError, extract_features,
    get_sorted_paths, mix_signal, subject_rng,
)

CLEAN = 'clean'
MIXED = 'mixed'
VARIANTS = (CLEAN, MIXED, *ANOMALY_KINDS)


class DaliaSource(DataSource):
    """PPG-DaLiA as ml.preprocessing wrote it to disk, optionally restricted to the
    windows a subject spent in one of ``activities``. """

    def __init__(self, data_root: Path, key: str = 'ppg-dalia',
                 activities: tuple[int, ...] | None = None):
        self.key = key
        self.data_root = data_root
        self.activities = activities
        self.clean_dir = data_root / CLEAN_SUBDIR
        self._mixed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._stats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def signal_dir(self, variant: str) -> Path:
        if variant == CLEAN:
            return self.clean_dir
        if variant in ANOMALY_KINDS:
            return self.data_root / ANOMALOUS_SUBDIR / variant
        raise ValueError(f"unknown signal variant {variant!r}, expected one of {VARIANTS}")

    def subject_ids(self) -> list[str]:
        dirs = get_sorted_paths(self.clean_dir)
        if not dirs:
            raise DatasetUnavailibleError(self.clean_dir)
        return [d.name for d in dirs]

    def _length(self, path: Path) -> int:
        return int(np.load(path, mmap_mode='r').shape[0])

    def n_windows(self, sid: str, window: int, shift: int) -> int:
        n_bvp = self._length(self.clean_dir / sid / 'bvp.npy')
        count = (n_bvp - window) // shift + 1 if n_bvp >= window else 0
        if window == BVP_WINDOW and shift == BVP_WINDOW:
            # The feature grid is also bounded by ACC, which runs at half the rate and
            # can end a window short (see extract_features' pairing of the two).
            n_acc = self._length(self.clean_dir / sid / 'acc.npy')
            count = min(count, (n_acc - ACC_WINDOW) // ACC_WINDOW + 1)
        return max(count, 0)

    def window_mask(self, sid: str, window: int, shift: int) -> np.ndarray:
        count = self.n_windows(sid, window, shift)
        if self.activities is None:
            return np.ones(count, dtype=bool)

        activity = np.load(self.clean_dir / sid / ACTIVITY_FILE)
        allowed = np.zeros(int(activity.max()) + 1, dtype=bool)
        for act in self.activities:
            if act < len(allowed):
                allowed[act] = True

        # Running count of disallowed samples, so a window is kept iff it contains none.
        excluded = np.concatenate([[0], np.cumsum(~allowed[activity])])
        starts = np.arange(count) * shift
        return (excluded[starts + window] - excluded[starts]) == 0

    # -- raw streams ---------------------------------------------------------

    def _mix(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        """The subject's mixed signal and its per-window labels, built once per source."""
        if sid not in self._mixed:
            self._mixed[sid] = mix_signal(self.signal(sid, CLEAN), subject_rng(sid))
        return self._mixed[sid]

    def signal(self, sid: str, variant: str) -> np.ndarray:
        if variant == MIXED:
            return self._mix(sid)[0]
        path = self.signal_dir(variant) / sid / 'bvp.npy'
        if not path.exists():
            raise DatasetUnavailibleError(self.signal_dir(variant))
        return np.load(path)

    def acc_signal(self, sid: str) -> np.ndarray:
        return np.load(self.clean_dir / sid / 'acc.npy')

    def _raw_windows(self, sid: str, variant: str, window: int,
                     shift: int) -> np.ndarray:
        """``(n, window)`` un-normalized BVP windows on the kept grid."""
        signal = self.signal(sid, variant)
        count = self.n_windows(sid, window, shift)
        if count == 0:
            return np.empty((0, window), dtype=np.float32)
        windows = sliding_window_view(signal, window)[::shift][:count]
        return windows[self.window_mask(sid, window, shift)].astype(np.float32)

    def raw_features(self, sid: str, variant: str) -> np.ndarray:
        bvp = self._raw_windows(sid, variant, BVP_WINDOW, BVP_WINDOW)
        if not len(bvp):
            return np.empty((0, N_FEATURES), dtype=np.float32)

        acc = self.acc_signal(sid)
        count = self.n_windows(sid, BVP_WINDOW, BVP_WINDOW)
        acc_windows = np.stack([acc[i * ACC_WINDOW:(i + 1) * ACC_WINDOW]
                                for i in range(count)])
        acc_windows = acc_windows[self.window_mask(sid, BVP_WINDOW, BVP_WINDOW)]

        return np.stack([extract_features(b, a) for b, a in zip(bvp, acc_windows)])

    def _clean_reference(self, sid: str, family: str) -> np.ndarray:
        if family == 'signal':
            return self.signal(sid, CLEAN).reshape(-1, 1)
        if family == 'features':
            return self.raw_features(sid, CLEAN)
        if family == 'descriptor':
            return window_descriptors(
                self._raw_windows(sid, CLEAN, BVP_WINDOW, BVP_WINDOW))
        raise ValueError(f"unknown normalization family {family!r}")

    def norm_stats(self, sid: str, family: str) -> tuple[np.ndarray, np.ndarray]:
        key = (sid, family)
        if key not in self._stats:
            reference = self._clean_reference(sid, family)
            self._stats[key] = (reference.mean(axis=0).astype(np.float32),
                                reference.std(axis=0).astype(np.float32) + 1e-6)
        return self._stats[key]

    def _normalize(self, sid: str, family: str, values: np.ndarray) -> np.ndarray:
        mean, std = self.norm_stats(sid, family)
        return ((values - mean) / std).astype(np.float32)

    # -- model-facing accessors ----------------------------------------------

    def signal_windows(self, sid: str, variant: str, window: int,
                       shift: int) -> np.ndarray:
        raw = self._raw_windows(sid, variant, window, shift).reshape(-1, window, 1)
        return self._normalize(sid, 'signal', raw)

    def features(self, sid: str, variant: str = MIXED) -> np.ndarray:
        return self._normalize(sid, 'features', self.raw_features(sid, variant))

    def descriptors(self, sid: str, variant: str = MIXED, window: int = BVP_WINDOW,
                    shift: int = BVP_WINDOW) -> np.ndarray:
        raw = window_descriptors(self._raw_windows(sid, variant, window, shift))
        return self._normalize(sid, 'descriptor', raw)

    def window_labels(self, sid: str) -> np.ndarray:
        labels = self._mix(sid)[1]
        count = min(len(labels), self.n_windows(sid, BVP_WINDOW, BVP_WINDOW))
        return labels[:count][self.window_mask(sid, BVP_WINDOW, BVP_WINDOW)[:count]]

    # -- int8 calibration ----------------------------------------------------

    def _sample(self, arrays: list[np.ndarray], per_subject: int) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        parts = [a[rng.choice(len(a), min(per_subject, len(a)), replace=False)]
                 for a in arrays if len(a)]
        if not parts:
            raise DatasetUnavailibleError(self.data_root)
        return np.concatenate(parts)

    def calibration_windows(self, window: int, shift: int,
                            per_subject: int = 10) -> np.ndarray:
        return self._sample([self.signal_windows(sid, CLEAN, window, shift)
                             for sid in self.subject_ids()], per_subject)

    def calibration_features(self, per_subject: int = 10) -> np.ndarray:
        return self._sample([self.features(sid, CLEAN) for sid in self.subject_ids()],
                            per_subject)

    def calibration_descriptors(self, window: int = BVP_WINDOW, shift: int = BVP_WINDOW,
                                per_subject: int = 10) -> np.ndarray:
        return self._sample([self.descriptors(sid, CLEAN, window, shift)
                             for sid in self.subject_ids()], per_subject)
