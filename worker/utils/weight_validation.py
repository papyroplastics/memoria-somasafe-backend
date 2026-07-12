import numpy as np

from common.db import GlobalWeights, ClientDeltaSubmission


def update_magnitude(delta: np.ndarray) -> float:
    return float(np.mean(delta ** 2))


def malformed_reason(submission: ClientDeltaSubmission,
                     expected_weight_count: int) -> str | None:
    if submission.weight_count != expected_weight_count:
        return (f"weight count {submission.weight_count} != "
                f"model's {expected_weight_count}")
    if len(submission.deltas) != expected_weight_count * 4:
        return (f"buffer size {len(submission.deltas)} != "
                f"{expected_weight_count} float32 weights")
    if not np.all(np.isfinite(np.frombuffer(submission.deltas, dtype=np.float32))):
        return "weights contain non-finite values"
    return None


def validate_submission(submission: ClientDeltaSubmission, expected_weight_count: int,
                        reference: GlobalWeights | None) -> str | None:
    reason = malformed_reason(submission, expected_weight_count)
    if reason is not None:
        return reason
    if reference is not None and reference.mse_threshold is not None:
        delta = np.frombuffer(submission.deltas, dtype=np.float32)
        error = update_magnitude(delta)
        if error > reference.mse_threshold:
            return (f"update magnitude {error:.6g} exceeds "
                    f"threshold {reference.mse_threshold:.6g}")
    return None


def filter_outliers(vectors: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    if len(vectors) < 3:
        return np.ones(len(vectors), dtype=bool)
    distances = np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
    std = float(distances.std())
    if std < 1e-12:
        return np.ones(len(vectors), dtype=bool)
    return (distances - distances.mean()) / std <= z_threshold


def compute_mse_threshold(mses: list[float], margin: float = 2.0) -> float | None:
    threshold = max(mses) * margin
    return threshold if threshold > 0 else None
