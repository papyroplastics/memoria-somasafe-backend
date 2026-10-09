from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto

import numpy as np
from celery.utils.log import get_task_logger
from sqlalchemy import delete, func, select, update
from sqlmodel import Session

from common.celery_tasks import SECURE_SUM_TASK, SECURE_SWEEP_TASK
from common.config import (
    SECURE_SESSION_MIN_MEMBERS,
    SECURE_SESSION_OPEN_FAIL_TIMEOUT_SECONDS,
    SECURE_SESSION_OPEN_SEAL_TIMEOUT_SECONDS,
    SECURE_SESSION_SEALED_FAIL_TIMEOUT_SECONDS,
    WORKER_REAP_AFTER_SECONDS,
)
from common.db import (
    ModelVersion,
    SecurePartial,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    engine,
    get_latest_weights,
    seal_session,
    utcnow,
)
from common.ratelimit import clear_user_limits
from worker.celery_app import app
from worker.compute import (
    Action,
    SessionState,
    SweepAction,
    SweepPolicy,
    secure_session_mean,
    sweep_actions,
)
from worker.metrics import Timer, write

log = get_task_logger(__name__)

POLICY = SweepPolicy(
    min_members=SECURE_SESSION_MIN_MEMBERS,
    open_seal_timeout=SECURE_SESSION_OPEN_SEAL_TIMEOUT_SECONDS,
    open_fail_timeout=SECURE_SESSION_OPEN_FAIL_TIMEOUT_SECONDS,
    sealed_fail_timeout=SECURE_SESSION_SEALED_FAIL_TIMEOUT_SECONDS,
    summing_timeout=WORKER_REAP_AFTER_SECONDS,
)


class SecureSumStage(StrEnum):
    read = auto()
    ring_sum = auto()
    commit = auto()


class SecureSumOutcome(StrEnum):
    summed = auto()
    skipped_not_sealed = auto()
    skipped_not_summing = auto()
    stale_base = auto()
    failed = auto()


@dataclass
class SecureSumRecord:
    session_id: int
    outcome: SecureSumOutcome | None = None
    model_key: str | None = None
    base_weights_id: int | None = None
    members: int | None = None


def _vector(masked: bytes | None) -> np.ndarray:
    if masked is None:
        raise ValueError("a member has not submitted")
    return np.frombuffer(masked, dtype="<u4")


def _sum(session_id: int, timer: Timer[SecureSumStage], record: SecureSumRecord) -> None:
    with Session(engine) as session:
        with timer(SecureSumStage.read):
            current = session.get(SecureSession, session_id)
            record.model_key = current.model_key
            record.base_weights_id = current.base_weights_id
            record.members = current.member_count
            active = get_latest_weights(session, current.model_key)
            if active is None or active.id != current.base_weights_id:
                SecureSession.transition(session, session_id, SecureSessionStatus.summing,
                                         SecureSessionStatus.failed, finished_at=utcnow())
                session.commit()
                record.outcome = SecureSumOutcome.stale_base
                return
            weight_count = session.get(ModelVersion, current.version_id).weight_count

        with timer(SecureSumStage.ring_sum):
            rows = session.execute(
                select(SecureSessionMember.masked)  # type: ignore
                .where(SecureSessionMember.session_id == session_id)
                .execution_options(yield_per=8)).scalars()
            mean = secure_session_mean((_vector(blob) for blob in rows), weight_count,
                                       current.scale, current.member_count,
                                       current.clip_bound)

        with timer(SecureSumStage.commit):
            session.add(SecurePartial(session_id=session_id,
                                      mean=mean.astype(np.float32).tobytes()))
            session.execute(update(SecureSessionMember)
                            .where(SecureSessionMember.session_id == session_id)  # type: ignore
                            .values(masked=None))
            if not SecureSession.transition(session, session_id, SecureSessionStatus.summing,
                                            SecureSessionStatus.summed, finished_at=utcnow()):
                session.rollback()
                record.outcome = SecureSumOutcome.skipped_not_summing
                return
            session.commit()
    record.outcome = SecureSumOutcome.summed


@app.task(name=SECURE_SUM_TASK, ignore_result=True)
def secure_session_sum(session_id: int) -> None:
    timer = Timer(*SecureSumStage)
    record = SecureSumRecord(session_id)
    if not SecureSession.claim(session_id, SecureSessionStatus.sealed,
                               SecureSessionStatus.summing, summing_at=utcnow()):
        record.outcome = SecureSumOutcome.skipped_not_sealed
    else:
        try:
            _sum(session_id, timer, record)
        except Exception:
            log.exception("secure session %s failed", session_id)
            record.outcome = SecureSumOutcome.failed
            SecureSession.claim(session_id, SecureSessionStatus.summing,
                                SecureSessionStatus.failed, finished_at=utcnow())
    write(SECURE_SUM_TASK, record, timer)


def _read_sweep() -> tuple[list[SessionState], dict[str, int | None]]:
    with Session(engine) as session:
        rows = session.execute(
            select(SecureSession.id, SecureSession.model_key, SecureSession.status,
                   SecureSession.base_weights_id,
                   func.count(SecureSessionMember.user_id),
                   func.count(SecureSessionMember.masked),
                   SecureSession.member_count, SecureSession.created_at,
                   SecureSession.sealed_at, SecureSession.summing_at)
            .outerjoin(SecureSessionMember,
                       SecureSessionMember.session_id == SecureSession.id)  # type: ignore[arg-type]
            .where(SecureSession.status.in_((SecureSessionStatus.open,  # type: ignore[attr-defined]
                                             SecureSessionStatus.sealed,
                                             SecureSessionStatus.summing)))
            .group_by(SecureSession.id)).all()
        sessions = [SessionState(*row) for row in rows]
        active = {}
        for key in {s.model_key for s in sessions}:
            weights = get_latest_weights(session, key)
            active[key] = weights.id if weights is not None else None
    return sessions, active


def _fail(session: Session, action: SweepAction, now: datetime) -> list[int]:
    if not SecureSession.transition(session, action.session_id, action.frm,
                                    SecureSessionStatus.failed, finished_at=now):
        return []
    released = session.execute(
        delete(SecureSessionMember)
        .where(SecureSessionMember.session_id == action.session_id)  # type: ignore
        .returning(SecureSessionMember.user_id, SecureSessionMember.submitted_at)).all()
    return [user_id for user_id, submitted_at in released
            if action.frm is SecureSessionStatus.open or submitted_at is not None]


@app.task(name=SECURE_SWEEP_TASK, ignore_result=True)
def secure_session_sweep() -> None:
    now = utcnow()
    sessions, active = _read_sweep()
    dispatch: list[int] = []

    for action in sweep_actions(sessions, active, now, POLICY):
        if action.action is Action.dispatch:
            dispatch.append(action.session_id)
            continue
        cleared: list[int] = []
        with Session(engine) as session:
            if action.action is Action.seal:
                seal_session(session, action.session_id)
            elif action.action is Action.retry:
                SecureSession.transition(session, action.session_id, action.frm,
                                         SecureSessionStatus.sealed)
            else:
                cleared = _fail(session, action, now)
            session.commit()
        if cleared:
            clear_user_limits(action.model_key, cleared)

    for session_id in dispatch:
        secure_session_sum.delay(session_id)
