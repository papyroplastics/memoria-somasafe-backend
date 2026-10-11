from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from celery.utils.log import get_task_logger
from sqlalchemy import ColumnElement, Select, delete, func, select, update
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
    GlobalWeights,
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
from worker.compute import block_size, dense_update
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
    skipped_stale = auto()
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
    cohort: int | None = None
    submissions: int | None = None
    blocks: int | None = None


@dataclass(frozen=True)
class RoundPlan:
    version_id: int
    fingerprint: str
    contract_version: int
    reference_id: int
    weights: np.ndarray
    secure: bool


@dataclass(frozen=True)
class RoundRows:
    count: int
    submissions: int
    blob: ColumnElement
    query: Select


def _dense_rows(session: Session, reference_id: int) -> RoundRows:
    where = (ClientDeltaSubmission.base_weights_id == reference_id,
             ClientDeltaSubmission.valid == True,  # noqa: E712
             ClientDeltaSubmission.deltas.is_not(None))  # type: ignore[union-attr]
    count = session.execute(select(func.count()).select_from(ClientDeltaSubmission)
                            .where(*where)).scalar_one()
    return RoundRows(count, count, ClientDeltaSubmission.deltas,  # type: ignore[arg-type]
                     select(ClientDeltaSubmission).where(*where))


def _secure_rows(session: Session, reference_id: int) -> RoundRows:
    where = (SecureSession.base_weights_id == reference_id,
             SecureSession.status == SecureSessionStatus.summed)
    count, members = session.execute(
        select(func.count(), func.coalesce(func.sum(SecureSession.member_count), 0))
        .where(*where)).one()
    return RoundRows(count, members, SecurePartial.mean,  # type: ignore[arg-type]
                     select(SecurePartial)
                     .join(SecureSession, SecureSession.id == SecurePartial.session_id)  # type: ignore[arg-type]
                     .where(*where))


def _read_block(session: Session, rows: RoundRows, start: int, stop: int) -> np.ndarray:
    block = np.empty((rows.count, stop - start), dtype=np.float32)
    query = rows.query.with_only_columns(
        func.substring(rows.blob, start * 4 + 1, (stop - start) * 4))
    filled = 0
    for blob in session.execute(query.execution_options(yield_per=256)).scalars():
        block[filled] = np.frombuffer(blob, dtype=np.float32)
        filled += 1
    if filled != rows.count:
        raise RuntimeError(f"block read {filled} of {rows.count} rows")
    return block


def _set_closed(weights_id: int, closed: bool) -> None:
    with Session(engine) as session:
        session.execute(update(GlobalWeights)
                        .where(GlobalWeights.id == weights_id)  # type: ignore[arg-type]
                        .values(closed=closed))
        session.commit()


def _round_base(key: str, expected_id: int | None,
                record: AggregationRecord) -> tuple[int, int] | None:
    with Session(engine) as session:
        latest = get_latest_version(session, key)
        reference = get_version_weights(session, latest.id) if latest is not None else None
        if reference is None:
            record.outcome = AggregationOutcome.skipped_unavailable
            return None
        record.base_weights_id = reference.id
        if expected_id is not None and reference.id != expected_id:
            record.outcome = AggregationOutcome.skipped_stale
            return None
        return latest.id, reference.id


def _read_round(key: str, expected_id: int | None,
                timer: Timer[AggregationStage], record: AggregationRecord) -> RoundPlan | None:
    with timer(AggregationStage.metadata):
        base = _round_base(key, expected_id, record)
        if base is None:
            return None
        version_id, reference_id = base
        _set_closed(reference_id, True)

    with Session(engine) as session:
        session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        with timer(AggregationStage.metadata):
            latest = session.get(ModelVersion, version_id)
            reference = session.get(GlobalWeights, reference_id)
            secure = latest.submission_type is SubmissionType.secure
            rows = (_secure_rows if secure else _dense_rows)(session, reference_id)
            record.cohort = rows.count
            record.submissions = rows.submissions

        if rows.submissions < FED_MIN_SUBMISSIONS or rows.count == 0:
            _set_closed(reference_id, False)
            record.outcome = AggregationOutcome.skipped_min_submissions
            return None

        reference_weights = np.frombuffer(decompress(reference.weights), dtype=np.float32)
        weights = np.empty_like(reference_weights)
        size = block_size(rows.count, FED_AGG_MEMORY_BYTES)
        starts = range(0, latest.weight_count, size)
        record.blocks = len(starts)
        for start in starts:
            stop = min(start + size, latest.weight_count)
            with timer(AggregationStage.blob_fetch):
                block = _read_block(session, rows, start, stop)
            with timer(AggregationStage.trimmed_mean):
                weights[start:stop] = dense_update(reference_weights[start:stop], block,
                                                   FED_TRIM_RATIO)

        return RoundPlan(latest.id, latest.fingerprint, latest.contract_version,
                         reference_id, weights, secure)


def _aggregate(key: str, expected_id: int | None,
               timer: Timer[AggregationStage | Restore | QuantizedBake | TrainableBake],
               record: AggregationRecord) -> None:
    plan = _read_round(key, expected_id, timer, record)
    if plan is None:
        return

    with timer(AggregationStage.runtime):
        rt = runtime.get(key)
    if rt.fingerprint != plan.fingerprint:
        record.outcome = AggregationOutcome.skipped_unavailable
        return

    restore(rt.model, plan.weights, timer)
    try:
        quantized = bake_quantized(rt.model, rt.rep_dataset, plan.contract_version, timer)
        trainable = bake_trainable(rt.model, plan.contract_version, timer)
    except Exception:
        log.exception("artifact export for %s failed", key)
        record.outcome = AggregationOutcome.export_failed
        return
    with timer(AggregationStage.compress_weights):
        weights = compress_weights(plan.weights)

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
def federated_aggregation(model_key: str, base_weights_id: int | None = None) -> dict[str, object]:
    timer = Timer(*STAGES)
    record = AggregationRecord(model_key)
    with model_lock(f"agg:{model_key}", FED_LOCK_TTL_SECONDS) as held:
        if not held:
            record.outcome = AggregationOutcome.skipped_locked
        else:
            try:
                _aggregate(model_key, base_weights_id, timer, record)
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
