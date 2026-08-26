from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common.config import SEED

from .common import DataSource
from ..preprocessing import (
    ACC_WINDOW, ACTIVITY_FILE, ANOMALOUS_SUBDIR, ANOMALY_KINDS, BVP_WINDOW,
    CLEAN_SUBDIR, FEATURE_STATS_FILE, MIXED_FEATURE_SUBDIR, MIXED_SUBDIR,
    NORM_PARAMS_FILE, DatasetUnavailibleError, get_sorted_paths,
)

CLEAN = 'clean'
MIXED = 'mixed'
VARIANTS = (CLEAN, MIXED, *ANOMALY_KINDS)


class DaliaSource(DataSource):
    """PPG-DaLiA as ml.preprocessing wrote it to disk, optionally restricted to the
    windows a subject spent in one of ``activities``.

    The filter is strict: a window survives only if every one of its samples carries an
    allowed activity id, so windows straddling an activity change — and the transient
    periods between activities (id 0) — are dropped. It is derived from the single
    per-sample activity track stored next to bvp.npy, so the same window index means the
    same eight seconds on every grid and for every variant.
    """

    def __init__(self, data_root: Path, key: str = 'ppg-dalia',
                 activities: tuple[int, ...] | None = None):
        self.key = key
        self.data_root = data_root
        self.activities = activities
        self.clean_dir = data_root / CLEAN_SUBDIR
        self.feature_dir = data_root / MIXED_FEATURE_SUBDIR

    def signal_dir(self, variant: str) -> Path:
        if variant == CLEAN:
            return self.clean_dir
        if variant == MIXED:
            return self.data_root / MIXED_SUBDIR
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
            # can end a window short (see preprocessing.build_feature_dataset).
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

    def signal_windows(self, sid: str, variant: str, window: int,
                       shift: int) -> np.ndarray:
        path = self.signal_dir(variant) / sid / 'bvp.npy'
        if not path.exists():
            raise DatasetUnavailibleError(self.signal_dir(variant))
        signal = np.load(path)
        count = self.n_windows(sid, window, shift)
        if count == 0:
            return np.empty((0, window, 1), dtype=np.float32)
        windows = sliding_window_view(signal, window)[::shift][:count]
        kept = windows[self.window_mask(sid, window, shift)]
        return kept.reshape(-1, window, 1).astype(np.float32)

    def _feature_arrays(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        subject_dir = self.feature_dir / sid
        if not subject_dir.is_dir():
            raise DatasetUnavailibleError(self.feature_dir)
        x = np.load(subject_dir / 'features.npy').astype(np.float32)
        y = np.load(subject_dir / 'labels.npy').astype(np.float32).reshape(-1, 1)
        count = min(len(x), len(y), self.n_windows(sid, BVP_WINDOW, BVP_WINDOW))
        mask = self.window_mask(sid, BVP_WINDOW, BVP_WINDOW)[:count]
        return x[:count][mask], y[:count][mask]

    def features(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        return self._feature_arrays(sid)

    def window_labels(self, sid: str) -> np.ndarray:
        return self._feature_arrays(sid)[1].reshape(-1)

    def signal_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        path = self.clean_dir / NORM_PARAMS_FILE
        if not path.exists():
            raise DatasetUnavailibleError(self.clean_dir)
        params = np.load(path)
        return (np.array([params[0]], dtype=np.float32),
                np.array([params[1] + 1e-8], dtype=np.float32))

    def feature_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        path = self.feature_dir / FEATURE_STATS_FILE
        if not path.exists():
            raise DatasetUnavailibleError(self.feature_dir)
        stats = np.load(path)
        return stats[0].astype(np.float32), stats[1].astype(np.float32)

    def _sample(self, arrays: list[np.ndarray], per_subject: int) -> np.ndarray:
        """A few random rows from each subject rather than the head of each: every subject
        starts at rest, so a prefix would calibrate on an at-rest range and clip the rest."""
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
        return self._sample([self.features(sid)[0] for sid in self.subject_ids()],
                            per_subject)
