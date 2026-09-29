from dataclasses import dataclass
from enum import StrEnum, auto

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
from worker.baking import (
    QuantizedBake,
    Restore,
    TrainableBake,
    bake_quantized,
    bake_trainable,
    compress_weights,
    restore,
    store,
)
from worker.celery_app import app
from worker.compute import Action, RoundState, SweepPolicy, secure_update, sweep_actions
from worker.metrics import Timer, write

log = get_task_logger(__name__)

POLICY = SweepPolicy(
    min_members=SECURE_MIN_MEMBERS,
    target_members=SECURE_TARGET_MEMBERS,
    open_timeout=SECURE_ROUND_OPEN_TIMEOUT_SECONDS,
    seal_timeout=SECURE_ROUND_SEAL_TIMEOUT_SECONDS,
    aggregating_timeout=WORKER_REAP_AFTER_SECONDS,
)


class SecureAggregationStage(StrEnum):
    read = auto()
    runtime = auto()
    ring_sum = auto()
    compress_weights = auto()
    commit = auto()
    clear_limits = auto()


STAGES = (SecureAggregationStage.read, SecureAggregationStage.runtime,
          SecureAggregationStage.ring_sum, *Restore, *QuantizedBake, *TrainableBake,
          SecureAggregationStage.compress_weights, SecureAggregationStage.commit,
          SecureAggregationStage.clear_limits)


class SecureAggregationOutcome(StrEnum):
    aggregated = auto()
    skipped_not_sealed = auto()
    skipped_not_aggregating = auto()
    failed = auto()


@dataclass
class SecureAggregationRecord:
    round_id: int
    outcome: SecureAggregationOutcome | None = None
    model_key: str | None = None
    base_weights_id: int | None = None
    members: int | None = None


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


def _fail_round(round_id: int, frm: SecureRoundStatus) -> str | None:
    with Session(engine) as session:
        if not SecureRound.transition(session, round_id, frm, SecureRoundStatus.failed,
                                      finished_at=utcnow()):
            return None
        model_key = session.get(SecureRound, round_id).model_key
        session.commit()
        return model_key


def _aggregate(round_id: int,
               timer: Timer[SecureAggregationStage | Restore | QuantizedBake | TrainableBake],
               record: SecureAggregationRecord) -> None:
    with timer(SecureAggregationStage.read):
        plan = _read_round(round_id)
    record.model_key = plan.model_key
    record.base_weights_id = plan.base_id
    record.members = plan.member_count
    with timer(SecureAggregationStage.runtime):
        rt = runtime.get(plan.model_key)
    if rt.fingerprint != plan.fingerprint:
        raise ValueError("round version is no longer current")

    with timer(SecureAggregationStage.ring_sum):
        new_weights = secure_update(plan.reference, plan.vectors, plan.scale,
                                    plan.member_count, plan.clip_bound)
    restore(rt.model, new_weights, timer)
    try:
        quantized = bake_quantized(rt.model, rt.rep_dataset, plan.contract_version, timer)
        trainable = bake_trainable(rt.model, plan.contract_version, timer)
    except Exception as exc:
        raise ValueError(f"artifact export failed: {exc}") from exc
    with timer(SecureAggregationStage.compress_weights):
        weights = compress_weights(new_weights)

    try:
        with timer(SecureAggregationStage.commit), Session(engine) as session:
            store(session, plan.model_key, plan.version_id, plan.base_id, weights,
                  trainable, quantized)
            if not SecureRound.transition(session, round_id, SecureRoundStatus.aggregating,
                                          SecureRoundStatus.aggregated, finished_at=utcnow()):
                session.rollback()
                record.outcome = SecureAggregationOutcome.skipped_not_aggregating
                return
            session.commit()
    except IntegrityError:
        raise ValueError(f"base {plan.base_id} already aggregated") from None

    with timer(SecureAggregationStage.clear_limits):
        clear_model_limits(plan.model_key)
    record.outcome = SecureAggregationOutcome.aggregated


@app.task(name=SECURE_AGG_TASK, ignore_result=True)
def secure_aggregation(round_id: int) -> None:
    timer = Timer(*STAGES)
    record = SecureAggregationRecord(round_id)
    if not SecureRound.claim(round_id, SecureRoundStatus.sealed, SecureRoundStatus.aggregating,
                             aggregating_at=utcnow()):
        record.outcome = SecureAggregationOutcome.skipped_not_sealed
    else:
        try:
            _aggregate(round_id, timer, record)
        except Exception:
            log.exception("secure round %s failed", round_id)
            record.outcome = SecureAggregationOutcome.failed
            model_key = _fail_round(round_id, SecureRoundStatus.aggregating)
            if model_key is not None:
                clear_model_limits(model_key)
    write(SECURE_AGG_TASK, record, timer)


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


@app.task(name=SECURE_SWEEP_TASK, ignore_result=True)
def secure_round_sweep() -> None:
    now = utcnow()
    rounds, active = _read_sweep()
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
                                              SecureRoundStatus.failed, finished_at=now)
            session.commit()
        if done and action.action is not Action.seal:
            failed_models.add(action.model_key)

    for round_id in dispatch:
        secure_aggregation.delay(round_id)
    for key in failed_models:
        clear_model_limits(key)
