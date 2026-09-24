from datetime import datetime, timedelta

import numpy as np
import pytest

from common.db import SecureRoundStatus
from common.secure_agg import compute_scale, generate_keypair, mask_vector, quantize
from ml.aggregation import trimmed_mean_inplace
from worker.compute import (
    Action,
    RoundState,
    SweepPolicy,
    cohort_cap,
    dense_update,
    secure_update,
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


def test_secure_update_matches_plaintext_mean():
    rng = np.random.default_rng(1)
    n, m, clip, round_id = 3, 257, 1.0, 42
    scale = compute_scale(n, clip)
    keys = {uid: generate_keypair() for uid in (11, 12, 13)}
    roster = [(uid, pk) for uid, (_, pk) in keys.items()]
    deltas = {uid: rng.uniform(-clip, clip, m).astype(np.float32) for uid in keys}
    masked = {uid: mask_vector(quantize(deltas[uid], clip, scale), uid, roster, sk, round_id)
              for uid, (sk, _) in keys.items()}
    reference = rng.standard_normal(m).astype(np.float32)

    result = secure_update(reference, masked, scale, n, clip)

    expected = reference + np.mean(np.stack(list(deltas.values())), axis=0)
    assert float(np.max(np.abs(result - expected))) < 2.0 / scale + 1e-4


def test_secure_update_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        secure_update(np.zeros(4, dtype=np.float32), {1: np.zeros(3, dtype=np.uint32)},
                      1000, 1, 1.0)


def test_secure_update_rejects_implausible_mean():
    vectors = {uid: np.full(4, 2000, dtype=np.uint32) for uid in (1, 2, 3)}
    with pytest.raises(ValueError, match="sanity"):
        secure_update(np.zeros(4, dtype=np.float32), vectors, 1000, 3, 1.0)


NOW = datetime(2026, 1, 1, 12, 0, 0)
POLICY = SweepPolicy(min_members=3, target_members=5, open_timeout=60, seal_timeout=60,
                     aggregating_timeout=100)


def _round(status, members=0, submitted=0, member_count=None, age=0, base=1,
           sealed_age=None, aggregating_age=None) -> RoundState:
    ago = lambda seconds: None if seconds is None else NOW - timedelta(seconds=seconds)
    return RoundState(id=1, model_key="m", status=status, base_weights_id=base,
                      members=members, submitted=submitted, member_count=member_count,
                      created_at=ago(age), sealed_at=ago(sealed_age),
                      aggregating_at=ago(aggregating_age))


open_, sealed, aggregating = (SecureRoundStatus.open, SecureRoundStatus.sealed,
                              SecureRoundStatus.aggregating)


@pytest.mark.parametrize("state, expected", [
    (_round(open_, members=5, base=2), (Action.fail, "stale_base", 0)),
    (_round(open_, members=5), (Action.seal, "", 0)),
    (_round(open_, members=3), None),
    (_round(open_, members=3, age=60), (Action.seal, "", 0)),
    (_round(open_, members=2, age=60), (Action.fail, "open_timeout", 0)),
    (_round(open_, members=2), None),
    (_round(sealed, 3, 3, 3, sealed_age=1), (Action.dispatch, "", 0)),
    (_round(sealed, 3, 2, 3, sealed_age=1), None),
    (_round(sealed, 3, 2, 3, sealed_age=60), (Action.fail, "seal_timeout", 1)),
    (_round(sealed, 3, 3, 3, sealed_age=1, base=2), (Action.fail, "stale_base", 0)),
    (_round(aggregating, 3, 3, 3, aggregating_age=10), None),
    (_round(aggregating, 3, 3, 3, aggregating_age=100), (Action.fail, "worker_lost", 0)),
])
def test_sweep_actions(state, expected):
    actions = sweep_actions([state], {"m": 1}, NOW, POLICY)
    if expected is None:
        assert actions == []
    else:
        [action] = actions
        assert (action.action, action.reason, action.missing) == expected
        assert action.frm is state.status
