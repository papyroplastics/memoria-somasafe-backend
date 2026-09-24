from collections.abc import Sequence

import numpy as np


def trimmed_mean_inplace(stacked: np.ndarray, trim: float) -> np.ndarray:
    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    k = int(len(stacked) * trim)
    if k:
        stacked.sort(axis=0)
        stacked = stacked[k:len(stacked) - k]
    return stacked.mean(axis=0)


def trimmed_mean(vectors: Sequence[np.ndarray], trim: float) -> np.ndarray:
    return trimmed_mean_inplace(np.stack([np.asarray(vector) for vector in vectors]), trim)
