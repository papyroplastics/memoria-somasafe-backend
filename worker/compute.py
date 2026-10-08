from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from common.db import SecureSessionStatus
from common.secure_agg import RING_MODULUS, dequantize
from ml.aggregation import trimmed_mean_inplace


def cohort_cap(weight_count: int, memory_bytes: int) -> int:
    return max(1, memory_bytes // (weight_count * 4))


def dense_update(reference: np.ndarray, deltas: np.ndarray, trim: float) -> np.ndarray:
    return (reference + trimmed_mean_inplace(deltas, trim)).astype(np.float32)


def secure_session_mean(vectors: Iterable[np.ndarray], weight_count: int, scale: int,
                        member_count: int, clip_bound: float) -> np.ndarray:
    acc = np.zeros(weight_count, dtype=np.uint64)
    summed = 0
    for vector in vectors:
        if vector.size != weight_count:
            raise ValueError("member vector length mismatch")
        acc += vector
        summed += 1
    if summed != member_count:
        raise ValueError(f"{summed}/{member_count} vectors (masks only cancel with the full roster)")
    mean = dequantize((acc % RING_MODULUS).astype(np.uint32), scale, member_count)
    if not np.all(np.isfinite(mean)) or float(np.max(np.abs(mean))) > clip_bound * 1.001:
        raise ValueError("session sum failed sanity check (implausible mean delta)")
    return mean


@dataclass(frozen=True)
class SessionState:
    id: int
    model_key: str
    status: SecureSessionStatus
    base_weights_id: int
    members: int
    submitted: int
    member_count: int | None
    created_at: datetime
    sealed_at: datetime | None
    summing_at: datetime | None


@dataclass(frozen=True)
class SweepPolicy:
    min_members: int
    open_timeout: int
    seal_timeout: int
    summing_timeout: int


class Action(str, Enum):
    seal = "seal"
    dispatch = "dispatch"
    fail = "fail"
    retry = "retry"


@dataclass(frozen=True)
class SweepAction:
    session_id: int
    model_key: str
    action: Action
    frm: SecureSessionStatus
    reason: str = ""


def _elapsed(since: datetime | None, now: datetime, seconds: int) -> bool:
    return since is not None and now - since >= timedelta(seconds=seconds)


def sweep_actions(sessions: list[SessionState], active: dict[str, int | None],
                  now: datetime, policy: SweepPolicy) -> list[SweepAction]:
    actions = []
    for s in sessions:
        stale = active.get(s.model_key) != s.base_weights_id

        def act(action: Action, reason: str = "") -> None:
            actions.append(SweepAction(s.id, s.model_key, action, s.status, reason))

        if s.status is SecureSessionStatus.open:
            if stale:
                act(Action.fail, "stale_base")
            elif _elapsed(s.created_at, now, policy.open_timeout):
                if s.members >= policy.min_members:
                    act(Action.seal)
                else:
                    act(Action.fail, "open_timeout")
        elif s.status is SecureSessionStatus.sealed:
            expected = s.member_count or 0
            if stale:
                act(Action.fail, "stale_base")
            elif s.submitted >= expected:
                act(Action.dispatch)
            elif _elapsed(s.sealed_at, now, policy.seal_timeout):
                act(Action.fail, "seal_timeout")
        elif s.status is SecureSessionStatus.summing:
            if _elapsed(s.summing_at, now, policy.summing_timeout):
                act(Action.retry, "worker_lost")
    return actions
