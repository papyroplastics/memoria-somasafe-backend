from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from celery.utils.log import get_task_logger
from sqlalchemy import Select, delete, select
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from common.celery_tasks import FED_AGG_TASK, FED_DISPATCH_TASK
from common.compression import decompress
from common.config import (
    FED_AGG_MEMORY_BYTES,
    FED_LOCK_TTL_SECONDS,
    FED_MIN_SUBMISSIONS,
    FED_TRIM_RATIO,
)
from common.db import (
    ClientDeltaSubmission,
    ModelVersion,
    SecurePartial,
    SecureSession,
    SecureSessionStatus,
    SubmissionType,
    engine,
    get_latest_version,
    get_version_weights,
)
from common.ratelimit import clear_model_limits
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
from worker.compute import cohort_cap, dense_update
from worker.locking import model_lock
from worker.metrics import Timer, write

log = get_task_logger(__name__)


class AggregationStage(StrEnum):
    metadata = auto()
    blob_fetch = auto()
    runtime = auto()
    trimmed_mean = auto()
    compress_weights = auto()
    commit = auto()
    clear_limits = auto()


STAGES = (AggregationStage.metadata, AggregationStage.blob_fetch, AggregationStage.runtime,
          AggregationStage.trimmed_mean, *Restore, *QuantizedBake, *TrainableBake,
          AggregationStage.compress_weights, AggregationStage.commit,
          AggregationStage.clear_limits)


class AggregationOutcome(StrEnum):
    aggregated = auto()
    skipped_locked = auto()
    skipped_unavailable = auto()
    skipped_min_submissions = auto()
    export_failed = auto()
    duplicate_round = auto()
    failed = auto()


@dataclass
class AggregationRecord:
    model_key: str
    outcome: AggregationOutcome | None = None
    base_weights_id: int | None = None
    cap: int | None = None
    cohort: int | None = None
    submissions: int | None = None


@dataclass(frozen=True)
class RoundPlan:
    version_id: int
    fingerprint: str
    contract_version: int
    reference_id: int
    reference: np.ndarray
    deltas: np.ndarray
    secure: bool


def _dense_rows(session: Session, reference_id: int, cap: int) -> tuple[int, int, Select]:
    ids = list(session.execute(
        select(ClientDeltaSubmission.id)  # type: ignore
        .where(ClientDeltaSubmission.base_weights_id == reference_id,
               ClientDeltaSubmission.valid == True)
        .order_by(ClientDeltaSubmission.created_at.desc())  # type: ignore
        .limit(cap)).scalars())
    return len(ids), len(ids), (select(ClientDeltaSubmission.deltas)  # type: ignore
                                .where(ClientDeltaSubmission.id.in_(ids)))  # type: ignore


def _secure_rows(session: Session, reference_id: int, cap: int) -> tuple[int, int, Select]:
    rows = session.execute(
        select(SecureSession.id, SecureSession.member_count)
        .where(SecureSession.base_weights_id == reference_id,
               SecureSession.status == SecureSessionStatus.summed)
        .order_by(SecureSession.finished_at.desc())  # type: ignore
        .limit(cap)).all()
    ids = [session_id for session_id, _ in rows]
    return len(ids), sum(members for _, members in rows), (
        select(SecurePartial.mean)  # type: ignore
        .where(SecurePartial.session_id.in_(ids)))  # type: ignore


def _read_round(key: str, timer: Timer[AggregationStage],
                record: AggregationRecord) -> RoundPlan | None:
    with Session(engine) as session:
        with timer(AggregationStage.metadata):
            latest = get_latest_version(session, key)
            if latest is None:
                record.outcome = AggregationOutcome.skipped_unavailable
                return None
            reference = get_version_weights(session, latest.id)
            if reference is None:
                record.outcome = AggregationOutcome.skipped_unavailable
                return None
            record.base_weights_id = reference.id

            secure = latest.submission_type is SubmissionType.secure
            cap = cohort_cap(latest.weight_count, FED_AGG_MEMORY_BYTES)
            cohort, submissions, blobs = (_secure_rows if secure else _dense_rows)(
                session, reference.id, cap)
            record.cap = cap
            record.cohort = cohort
            record.submissions = submissions

        if submissions < FED_MIN_SUBMISSIONS:
            record.outcome = AggregationOutcome.skipped_min_submissions
            return None

        with timer(AggregationStage.blob_fetch):
            deltas = np.empty((cohort, latest.weight_count), dtype=np.float32)
            rows = session.execute(blobs.execution_options(yield_per=64)).scalars()
            filled = 0
            for blob in rows:
                deltas[filled] = np.frombuffer(blob, dtype=np.float32)
                filled += 1
            deltas = deltas[:filled]
            weights = np.frombuffer(decompress(reference.weights), dtype=np.float32)
        record.cohort = filled

        return RoundPlan(latest.id, latest.fingerprint, latest.contract_version,
                         reference.id, weights, deltas, secure)


def _aggregate(key: str,
               timer: Timer[AggregationStage | Restore | QuantizedBake | TrainableBake],
               record: AggregationRecord) -> None:
    plan = _read_round(key, timer, record)
    if plan is None:
        return

    with timer(AggregationStage.runtime):
        rt = runtime.get(key)
    if rt.fingerprint != plan.fingerprint:
        record.outcome = AggregationOutcome.skipped_unavailable
        return

    with timer(AggregationStage.trimmed_mean):
        new_weights = dense_update(plan.reference, plan.deltas, FED_TRIM_RATIO)
    restore(rt.model, new_weights, timer)
    try:
        quantized = bake_quantized(rt.model, rt.rep_dataset, plan.contract_version, timer)
        trainable = bake_trainable(rt.model, plan.contract_version, timer)
    except Exception:
        log.exception("artifact export for %s failed", key)
        record.outcome = AggregationOutcome.export_failed
        return
    with timer(AggregationStage.compress_weights):
        weights = compress_weights(new_weights)

    try:
        with timer(AggregationStage.commit), Session(engine) as session:
            store(session, key, plan.version_id, plan.reference_id, weights, trainable, quantized)
            if plan.secure:
                session.execute(delete(SecurePartial).where(
                    SecurePartial.session_id.in_(  # type: ignore[attr-defined]
                        select(SecureSession.id)
                        .where(SecureSession.base_weights_id == plan.reference_id))))
            session.commit()
    except IntegrityError:
        record.outcome = AggregationOutcome.duplicate_round
        return

    with timer(AggregationStage.clear_limits):
        clear_model_limits(key)
    record.outcome = AggregationOutcome.aggregated


@app.task(name=FED_AGG_TASK)
def federated_aggregation(model_key: str) -> dict[str, object]:
    timer = Timer(*STAGES)
    record = AggregationRecord(model_key)
    with model_lock(f"agg:{model_key}", FED_LOCK_TTL_SECONDS) as held:
        if not held:
            record.outcome = AggregationOutcome.skipped_locked
        else:
            try:
                _aggregate(model_key, timer, record)
            except Exception:
                log.exception("aggregation for %s failed", model_key)
                record.outcome = AggregationOutcome.failed
    return write(FED_AGG_TASK, record, timer)


@app.task(name=FED_DISPATCH_TASK)
def dispatch_aggregation() -> list[str]:
    with Session(engine) as session:
        keys = list(session.execute(
            select(ModelVersion.model_key).distinct()).scalars())
    for key in keys:
        federated_aggregation.delay(key)
    return keys
