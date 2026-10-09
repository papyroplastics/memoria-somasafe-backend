from datetime import datetime, timedelta

import numpy as np
import pytest

from common.db import SecureSessionStatus
from common.secure_agg import compute_scale, generate_keypair, mask_vector, quantize
from ml.aggregation import trimmed_mean_inplace
from worker.compute import (
    Action,
    SessionState,
    SweepPolicy,
    cohort_cap,
    dense_update,
    secure_session_mean,
    sweep_actions,
)


def _reference_trimmed_mean(matrix: np.ndarray, trim: float) -> np.ndarray:
    k = int(len(matrix) * trim)
    ordered = np.sort(matrix, axis=0)
    return ordered[k:len(ordered) - k].mean(axis=0)


@pytest.mark.parametrize("trim", [0.0, 0.1, 0.2, 0.4])
def test_trimmed_mean_matches_reference(trim):
    matrix = np.random.default_rng(0).standard_normal((23, 101)).astype(np.float32)
    expected = _reference_trimmed_mean(matrix.copy(), trim)
    np.testing.assert_allclose(trimmed_mean_inplace(matrix, trim), expected, atol=1e-6)


def test_trimmed_mean_drops_outliers():
    matrix = np.ones((10, 4), dtype=np.float32)
    matrix[0] = 1e6
    matrix[1] = -1e6
    np.testing.assert_allclose(trimmed_mean_inplace(matrix, 0.1), np.ones(4))


def test_trimmed_mean_rejects_bad_trim():
    with pytest.raises(ValueError):
        trimmed_mean_inplace(np.zeros((3, 3), dtype=np.float32), 0.5)


def test_cohort_cap():
    assert cohort_cap(109386, 500 * 1024 * 1024) == 500 * 1024 * 1024 // (109386 * 4)
    assert cohort_cap(10**9, 1024) == 1


def test_dense_update():
    reference = np.arange(5, dtype=np.float32)
    deltas = np.tile(np.float32(0.5), (4, 5))
    np.testing.assert_allclose(dense_update(reference, deltas, 0.2), reference + 0.5)


def _masked_cohort(n: int, m: int, clip: float, seed: int):
    rng = np.random.default_rng(seed)
    scale = compute_scale(n, clip)
    keys = {uid: generate_keypair() for uid in range(11, 11 + n)}
    roster = [(uid, pk) for uid, (_, pk) in keys.items()]
    deltas = {uid: rng.uniform(-clip, clip, m).astype(np.float32) for uid in keys}
    masked = [mask_vector(quantize(deltas[uid], clip, scale), uid, roster, sk, 42)
              for uid, (sk, _) in keys.items()]
    return deltas, masked, scale


def test_secure_session_mean_matches_plaintext_mean():
    n, m, clip = 4, 257, 1.0
    deltas, masked, scale = _masked_cohort(n, m, clip, seed=1)
    mean = secure_session_mean(iter(masked), m, scale, n, clip)
    expected = np.mean(np.stack(list(deltas.values())), axis=0)
    assert float(np.max(np.abs(mean - expected))) < 2.0 / scale + 1e-4


def test_secure_session_mean_rejects_missing_member():
    n, m, clip = 3, 16, 1.0
    _, masked, scale = _masked_cohort(n, m, clip, seed=2)
    with pytest.raises(ValueError, match="full roster"):
        secure_session_mean(iter(masked[:-1]), m, scale, n, clip)


def test_secure_session_mean_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        secure_session_mean([np.zeros(3, dtype=np.uint32)], 4, 1000, 1, 1.0)


def test_secure_session_mean_rejects_implausible_mean():
    vectors = [np.full(4, 2000, dtype=np.uint32) for _ in range(3)]
    with pytest.raises(ValueError, match="sanity"):
        secure_session_mean(vectors, 4, 1000, 3, 1.0)


NOW = datetime(2026, 1, 1, 12, 0, 0)
POLICY = SweepPolicy(min_members=3, open_seal_timeout=30, open_fail_timeout=60,
                     sealed_fail_timeout=60, summing_timeout=100)


def _session(status, members=0, submitted=0, member_count=None, age=0, base=1,
             sealed_age=None, summing_age=None) -> SessionState:
    ago = lambda seconds: None if seconds is None else NOW - timedelta(seconds=seconds)
    return SessionState(id=1, model_key="m", status=status, base_weights_id=base,
                        members=members, submitted=submitted, member_count=member_count,
                        created_at=ago(age), sealed_at=ago(sealed_age),
                        summing_at=ago(summing_age))


open_, sealed, summing = (SecureSessionStatus.open, SecureSessionStatus.sealed,
                          SecureSessionStatus.summing)


@pytest.mark.parametrize("state, expected", [
    (_session(open_, members=5, base=2), (Action.fail, "stale_base")),
    (_session(open_, members=5), None),
    (_session(open_, members=3, age=30), (Action.seal, "")),
    (_session(open_, members=3, age=29), None),
    (_session(open_, members=2, age=30), None),
    (_session(open_, members=2, age=60), (Action.fail, "open_timeout")),
    (_session(open_, members=2), None),
    (_session(sealed, 3, 3, 3, sealed_age=1), (Action.dispatch, "")),
    (_session(sealed, 3, 3, 3, sealed_age=600), (Action.dispatch, "")),
    (_session(sealed, 3, 2, 3, sealed_age=1), None),
    (_session(sealed, 3, 2, 3, sealed_age=60), (Action.fail, "seal_timeout")),
    (_session(sealed, 3, 3, 3, sealed_age=1, base=2), (Action.fail, "stale_base")),
    (_session(summing, 3, 3, 3, summing_age=10), None),
    (_session(summing, 3, 3, 3, summing_age=100), (Action.retry, "worker_lost")),
    (_session(summing, 3, 3, 3, summing_age=100, base=2), (Action.retry, "worker_lost")),
])
def test_sweep_actions(state, expected):
    actions = sweep_actions([state], {"m": 1}, NOW, POLICY)
    if expected is None:
        assert actions == []
    else:
        [action] = actions
        assert (action.action, action.reason) == expected
        assert action.frm is state.status
