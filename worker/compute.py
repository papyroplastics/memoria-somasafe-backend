from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from common.db import SecureRoundStatus
from common.secure_agg import dequantize, ring_sum
from ml.aggregation import trimmed_mean_inplace


def cohort_cap(weight_count: int, memory_bytes: int) -> int:
    return max(1, memory_bytes // (weight_count * 4))


def dense_update(reference: np.ndarray, deltas: np.ndarray, trim: float) -> np.ndarray:
    return (reference + trimmed_mean_inplace(deltas, trim)).astype(np.float32)


def secure_update(reference: np.ndarray, vectors: dict[int, np.ndarray], scale: int,
                  member_count: int, clip_bound: float) -> np.ndarray:
    for user_id, vector in vectors.items():
        if vector.size != reference.size:
            raise ValueError(f"member {user_id} vector length mismatch")
    mean_delta = dequantize(ring_sum(list(vectors.values())), scale, member_count)
    new_weights = (reference + mean_delta).astype(np.float32)
    if not np.all(np.isfinite(new_weights)) \
            or float(np.max(np.abs(mean_delta))) > clip_bound * 1.001:
        raise ValueError("aggregate failed sanity check (implausible mean delta)")
    return new_weights


@dataclass(frozen=True)
class RoundState:
    id: int
    model_key: str
    status: SecureRoundStatus
    base_weights_id: int
    members: int
    submitted: int
    member_count: int | None
    created_at: datetime
    sealed_at: datetime | None
    aggregating_at: datetime | None


@dataclass(frozen=True)
class SweepPolicy:
    min_members: int
    target_members: int
    open_timeout: int
    seal_timeout: int
    aggregating_timeout: int


class Action(str, Enum):
    seal = "seal"
    dispatch = "dispatch"
    fail = "fail"


@dataclass(frozen=True)
class SweepAction:
    round_id: int
    model_key: str
    action: Action
    frm: SecureRoundStatus
    reason: str = ""
    missing: int = 0


def _elapsed(since: datetime | None, now: datetime, seconds: int) -> bool:
    return since is not None and now - since >= timedelta(seconds=seconds)


def sweep_actions(rounds: list[RoundState], active: dict[str, int | None],
                  now: datetime, policy: SweepPolicy) -> list[SweepAction]:
    actions = []
    for r in rounds:
        stale = active.get(r.model_key) != r.base_weights_id

        def act(action: Action, reason: str = "", missing: int = 0) -> None:
            actions.append(SweepAction(r.id, r.model_key, action, r.status, reason, missing))

        if r.status is SecureRoundStatus.open:
            timed_out = _elapsed(r.created_at, now, policy.open_timeout)
            if stale:
                act(Action.fail, "stale_base")
            elif r.members >= policy.min_members \
                    and (r.members >= policy.target_members or timed_out):
                act(Action.seal)
            elif timed_out:
                act(Action.fail, "open_timeout")
        elif r.status is SecureRoundStatus.sealed:
            expected = r.member_count or 0
            if stale:
                act(Action.fail, "stale_base")
            elif r.submitted >= expected:
                act(Action.dispatch)
            elif _elapsed(r.sealed_at, now, policy.seal_timeout):
                act(Action.fail, "seal_timeout", expected - r.submitted)
        elif r.status is SecureRoundStatus.aggregating:
            if _elapsed(r.aggregating_at, now, policy.aggregating_timeout):
                act(Action.fail, "worker_lost")
    return actions
