from collections import Counter
from dataclasses import dataclass

import numpy as np
from celery.utils.log import get_task_logger
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from common.celery_tasks import SECURE_AGG_TASK, SECURE_SWEEP_TASK
from common.compression import decompress
from common.config import (
    SECURE_MIN_MEMBERS,
    SECURE_ROUND_OPEN_TIMEOUT_SECONDS,
    SECURE_ROUND_SEAL_TIMEOUT_SECONDS,
    SECURE_TARGET_MEMBERS,
    WORKER_REAP_AFTER_SECONDS,
)
from common.db import (
    GlobalWeights,
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    engine,
    get_latest_version,
    get_latest_weights,
    utcnow,
)
from common.ratelimit import clear_model_limits
from common.secure_round import seal_round
from worker import runtime
from worker.baking import bake, store
from worker.celery_app import app
from worker.compute import Action, RoundState, SweepPolicy, secure_update, sweep_actions
from worker.phases import Phases

log = get_task_logger(__name__)

POLICY = SweepPolicy(
    min_members=SECURE_MIN_MEMBERS,
    target_members=SECURE_TARGET_MEMBERS,
    open_timeout=SECURE_ROUND_OPEN_TIMEOUT_SECONDS,
    seal_timeout=SECURE_ROUND_SEAL_TIMEOUT_SECONDS,
    aggregating_timeout=WORKER_REAP_AFTER_SECONDS,
)

_SWEEP_COUNTERS = ("sealed", "dispatched", "failed_stale_base", "failed_open_timeout",
                   "failed_seal_timeout", "failed_worker_lost", "members_missing_at_timeout")


@dataclass(frozen=True)
class SecurePlan:
    model_key: str
    version_id: int
    fingerprint: str
    contract_version: int
    base_id: int
    reference: np.ndarray
    vectors: dict[int, np.ndarray]
    scale: int
    member_count: int
    clip_bound: float


def _read_round(round_id: int) -> SecurePlan:
    with Session(engine) as session:
        round = session.get(SecureRound, round_id)
        latest = get_latest_version(session, round.model_key)
        if latest is None or latest.id != round.version_id:
            raise ValueError("round version is no longer current")
        members = session.execute(
            select(SecureRoundMember).where(SecureRoundMember.round_id == round_id)
        ).scalars().all()
        vectors = {m.user_id: np.frombuffer(m.masked, dtype="<u4").astype(np.uint32)
                   for m in members if m.masked is not None}
        if len(vectors) != len(members) or len(members) != round.member_count:
            raise ValueError(f"{len(vectors)}/{round.member_count} members submitted "
                             f"(masks only cancel with the full roster)")
        base = session.get(GlobalWeights, round.base_weights_id)
        if base is None:
            raise ValueError("base weights missing")
        return SecurePlan(
            model_key=round.model_key, version_id=latest.id,
            fingerprint=latest.fingerprint, contract_version=latest.contract_version,
            base_id=base.id,
            reference=np.frombuffer(decompress(base.weights), dtype=np.float32),
            vectors=vectors, scale=round.scale, member_count=round.member_count,
            clip_bound=round.clip_bound,
        )


def _fail_round(round_id: int, frm: SecureRoundStatus, reason: str) -> str | None:
    with Session(engine) as session:
        if not SecureRound.transition(session, round_id, frm, SecureRoundStatus.failed,
                                      error=reason, finished_at=utcnow()):
            return None
        model_key = session.get(SecureRound, round_id).model_key
        session.commit()
        return model_key


def _result(round_id: int, outcome: str, detail: str, phases: Phases,
            members: int = 0) -> dict:
    log.info("round %s: %s (%s)", round_id, outcome, detail)
    return {"round_id": round_id, "outcome": outcome, "members": members,
            "detail": detail, "timings": phases.timings}


def _aggregate(round_id: int, phases: Phases) -> dict:
    with phases("read"):
        plan = _read_round(round_id)
    with phases("runtime"):
        rt = runtime.get(plan.model_key)
    if rt.fingerprint != plan.fingerprint:
        raise ValueError("round version is no longer current")

    with phases("ring_sum"):
        new_weights = secure_update(plan.reference, plan.vectors, plan.scale,
                                    plan.member_count, plan.clip_bound)
    try:
        baked = bake(rt, new_weights, plan.contract_version, phases)
    except Exception as exc:
        raise ValueError(f"artifact export failed: {exc}") from exc

    try:
        with phases("commit"), Session(engine) as session:
            store(session, plan.model_key, plan.version_id, plan.base_id, baked)
            if not SecureRound.transition(session, round_id, SecureRoundStatus.aggregating,
                                          SecureRoundStatus.aggregated, finished_at=utcnow()):
                session.rollback()
                return _result(round_id, "skipped_claimed", "round no longer aggregating",
                               phases)
            session.commit()
    except IntegrityError:
        raise ValueError(f"base {plan.base_id} already aggregated") from None

    with phases("clear_model_limits"):
        clear_model_limits(plan.model_key)
    return _result(round_id, "aggregated", f"{plan.member_count} members",
                   phases, plan.member_count)


@app.task(name=SECURE_AGG_TASK)
def secure_aggregation(round_id: int) -> dict:
    phases = Phases(task="secure_aggregation", round_id=round_id)
    if not SecureRound.claim(round_id, SecureRoundStatus.sealed, SecureRoundStatus.aggregating,
                             aggregating_at=utcnow()):
        return _result(round_id, "skipped_claimed", "round is not sealed", phases)
    try:
        return _aggregate(round_id, phases)
    except Exception as exc:
        model_key = _fail_round(round_id, SecureRoundStatus.aggregating, str(exc))
        if model_key is not None:
            clear_model_limits(model_key)
        return _result(round_id, "failed", str(exc), phases)


def _read_sweep() -> tuple[list[RoundState], dict[str, int | None]]:
    with Session(engine) as session:
        rows = session.execute(
            select(SecureRound.id, SecureRound.model_key, SecureRound.status,
                   SecureRound.base_weights_id,
                   func.count(SecureRoundMember.user_id),
                   func.count(SecureRoundMember.masked),
                   SecureRound.member_count, SecureRound.created_at,
                   SecureRound.sealed_at, SecureRound.aggregating_at)
            .outerjoin(SecureRoundMember,
                       SecureRoundMember.round_id == SecureRound.id)  # type: ignore[arg-type]
            .where(SecureRound.status.in_((SecureRoundStatus.open,  # type: ignore[attr-defined]
                                           SecureRoundStatus.sealed,
                                           SecureRoundStatus.aggregating)))
            .group_by(SecureRound.id)).all()
        rounds = [RoundState(*row) for row in rows]
        active = {}
        for key in {r.model_key for r in rounds}:
            weights = get_latest_weights(session, key)
            active[key] = weights.id if weights is not None else None
    return rounds, active


@app.task(name=SECURE_SWEEP_TASK)
def secure_round_sweep() -> dict[str, int]:
    now = utcnow()
    rounds, active = _read_sweep()
    counters = Counter({name: 0 for name in _SWEEP_COUNTERS})
    dispatch: list[int] = []
    failed_models: set[str] = set()

    for action in sweep_actions(rounds, active, now, POLICY):
        if action.action is Action.dispatch:
            dispatch.append(action.round_id)
            continue
        with Session(engine) as session:
            if action.action is Action.seal:
                done = seal_round(session, action.round_id) is not None
            else:
                done = SecureRound.transition(session, action.round_id, action.frm,
                                              SecureRoundStatus.failed, error=action.reason,
                                              finished_at=now)
            session.commit()
        if not done:
            continue
        if action.action is Action.seal:
            counters["sealed"] += 1
        else:
            counters[f"failed_{action.reason}"] += 1
            counters["members_missing_at_timeout"] += action.missing
            failed_models.add(action.model_key)

    for round_id in dispatch:
        secure_aggregation.delay(round_id)
        counters["dispatched"] += 1
    for key in failed_models:
        clear_model_limits(key)
    return dict(counters)
