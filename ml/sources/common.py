from abc import ABC, abstractmethod

import numpy as np


class DataSource(ABC):
    """Everything a consumer needs to read one dataset off disk, already through whatever
    load-time filter its DatasetSpec applies. Sources hand back numpy arrays, not tf.data:
    a filter is then a boolean mask, the scoring path in scripts.common.scoring consumes
    arrays directly, and ml.loading wraps them for the training loops.

    Every accessor is indexed by subject id and by a window grid, and every array a source
    returns for one (subject, grid) is aligned with every other: window i of
    ``signal_windows(sid, 'mixed', ...)`` is the window ``features(sid)[i]`` was extracted
    from and ``window_labels(sid)[i]`` labels."""

    key: str

    @abstractmethod
    def subject_ids(self) -> list[str]:
        """Subjects this source can serve, in a stable order."""

    @abstractmethod
    def n_windows(self, sid: str, window: int, shift: int) -> int:
        """Windows of ``window`` samples every ``shift`` this subject yields before
        filtering — the grid every accessor below is truncated to."""

    @abstractmethod
    def window_mask(self, sid: str, window: int, shift: int) -> np.ndarray:
        """Boolean mask over that grid: which windows this source keeps."""

    @abstractmethod
    def signal_windows(self, sid: str, variant: str, window: int,
                       shift: int) -> np.ndarray:
        """``(n, window, 1)`` float32 raw BVP windows. ``variant`` names the signal:
        'clean', 'mixed', or one of ``preprocessing.ANOMALY_KINDS``."""

    @abstractmethod
    def features(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        """``(n, N_FEATURES)`` raw feature vectors and their ``(n, 1)`` binary labels,
        on the non-overlapping window grid."""

    @abstractmethod
    def window_labels(self, sid: str) -> np.ndarray:
        """``(n,)`` binary anomaly truth on the non-overlapping window grid."""

    @abstractmethod
    def signal_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Global BVP (mean, std) baked into an autoencoder as its z-score constants."""

    @abstractmethod
    def feature_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Global per-feature (mean, std) baked into the feature model."""

    @abstractmethod
    def calibration_windows(self, window: int, shift: int,
                            per_subject: int = 10) -> np.ndarray:
        """A few random signal windows from each subject, for the int8 converter."""

    @abstractmethod
    def calibration_features(self, per_subject: int = 10) -> np.ndarray:
        """The same sample, as feature vectors."""
