import numpy as np

from common.db import GlobalWeights, WeightSubmission


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def malformed_reason(submission: WeightSubmission,
                     expected_param_count: int) -> str | None:
    """Reason the buffer itself is unusable (no artifact can even be produced
    from it), or None. Unlike the statistical gate below, a malformed submission
    is safe to report back to the client — it reveals nothing about the
    aggregation filters."""
    if submission.param_count != expected_param_count:
        return (f"parameter count {submission.param_count} != "
                f"model's {expected_param_count}")
    if len(submission.parameters) != expected_param_count * 4:
        return (f"buffer size {len(submission.parameters)} != "
                f"{expected_param_count} float32 parameters")
    if not np.all(np.isfinite(np.frombuffer(submission.parameters, dtype=np.float32))):
        return "parameters contain non-finite values"
    return None


def validate_submission(submission: WeightSubmission, expected_param_count: int,
                        reference: GlobalWeights | None) -> str | None:
    """Reason the submission is unusable for aggregation, or None if it is
    valid. The MSE gate compares against ``reference``'s parameters using the
    threshold computed by the aggregation round that produced it; without a
    reference or a threshold (no aggregation has happened yet) that check is
    skipped. The verdict is only ever cached on the row, never surfaced."""
    reason = malformed_reason(submission, expected_param_count)
    if reason is not None:
        return reason
    if reference is not None and reference.mse_threshold is not None:
        params = np.frombuffer(submission.parameters, dtype=np.float32)
        error = mse(params, np.frombuffer(reference.parameters, dtype=np.float32))
        if error > reference.mse_threshold:
            return (f"MSE against current global weights {error:.6g} exceeds "
                    f"threshold {reference.mse_threshold:.6g}")
    return None


def filter_outliers(vectors: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """Inlier mask over the rows of ``vectors`` (n, p): z-scores each row's L2
    distance from the element-wise mean and cuts everything above
    ``z_threshold``. With fewer than 3 rows (or near-identical distances) the
    statistic is meaningless, so everything is kept."""
    if len(vectors) < 3:
        return np.ones(len(vectors), dtype=bool)
    distances = np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
    std = float(distances.std())
    if std < 1e-12:
        return np.ones(len(vectors), dtype=bool)
    return (distances - distances.mean()) / std <= z_threshold


def compute_mse_threshold(mses: list[float], margin: float = 2.0) -> float | None:
    """Allowed submission error for the next round: ``margin`` times the worst
    deviation accepted this round. None when every accepted submission was
    identical to the reference (e.g. weights resubmitted untrained) — a zero
    threshold would reject any real update, so the next round skips the gate
    instead."""
    threshold = max(mses) * margin
    return threshold if threshold > 0 else None
