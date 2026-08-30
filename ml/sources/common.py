from abc import ABC, abstractmethod

import numpy as np


class DataSource(ABC):
    """Everything a consumer needs to read one dataset, already through whatever
    load-time filter its DatasetSpec applies. Sources hand back numpy arrays, not tf.data:
    a filter is then a boolean mask, the scoring path in scripts.common.scoring consumes
    arrays directly, and ml.loading wraps them for the training loops."""

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
    def signal(self, sid: str, variant: str) -> np.ndarray:
        """The subject's whole **raw** BVP stream for one variant, unwindowed and
        unfiltered — what a sensor would emit. Only the export scripts want this; a model
        is fed by ``signal_windows``."""

    @abstractmethod
    def acc_signal(self, sid: str) -> np.ndarray:
        """The subject's whole raw ACC magnitude stream. Anomalies are injected into BVP
        only, so there is one of these regardless of variant."""

    @abstractmethod
    def norm_stats(self, sid: str, family: str) -> tuple[np.ndarray, np.ndarray]:
        """The (mean, std) this source z-scores one subject's values with, for one value
        family ('signal', 'features', 'descriptor'). Every model-facing accessor already
        applies these; they are exposed because whoever feeds a model outside this
        pipeline — the device, through the export scripts — has to apply them itself."""

    @abstractmethod
    def raw_features(self, sid: str, variant: str) -> np.ndarray:
        """``(n, N_FEATURES)`` **un-normalized** feature vectors, on the non-overlapping
        window grid — the vectors as the firmware computes and reports them. Like
        ``signal``, this exists for the export scripts; a model is fed by ``features``."""

    @abstractmethod
    def signal_windows(self, sid: str, variant: str, window: int,
                       shift: int) -> np.ndarray:
        """``(n, window, 1)`` float32 per-subject-normalized BVP windows. ``variant``
        names the signal: 'clean', 'mixed', or one of ``preprocessing.ANOMALY_KINDS``."""

    @abstractmethod
    def features(self, sid: str, variant: str) -> np.ndarray:
        """``(n, N_FEATURES)`` per-subject-normalized feature vectors, on the
        non-overlapping window grid."""

    @abstractmethod
    def descriptors(self, sid: str, variant: str, window: int,
                    shift: int) -> np.ndarray:
        """``(n, N_DESCRIPTORS)`` per-subject-normalized spectral descriptors."""

    @abstractmethod
    def window_labels(self, sid: str) -> np.ndarray:
        """``(n,)`` binary anomaly truth for the mixed variant, on the non-overlapping
        window grid."""

    @abstractmethod
    def calibration_windows(self, window: int, shift: int,
                            per_subject: int = 10) -> np.ndarray:
        """A few normalized signal windows from each subject, for the int8 converter."""

    @abstractmethod
    def calibration_features(self, per_subject: int = 10) -> np.ndarray:
        """The same sample, as normalized feature vectors."""

    @abstractmethod
    def calibration_descriptors(self, window: int, shift: int,
                                per_subject: int = 10) -> np.ndarray:
        """The same sample, as normalized spectral descriptors."""
