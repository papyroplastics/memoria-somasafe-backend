from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from celery.utils.log import get_task_logger
from sqlalchemy import func, select
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

DENSE_TYPES = (SubmissionType.raw, SubmissionType.quantize)


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
    users: int | None = None
    on_base: int | None = None
    cap: int | None = None
    cohort: int | None = None


@dataclass(frozen=True)
class RoundPlan:
    version_id: int
    fingerprint: str
    contract_version: int
    reference_id: int
    reference: np.ndarray
    deltas: np.ndarray


def _read_round(key: str, timer: Timer[AggregationStage],
                record: AggregationRecord) -> RoundPlan | None:
    with Session(engine) as session:
        with timer(AggregationStage.metadata):
            latest = get_latest_version(session, key)
            if latest is None or latest.submission_type not in DENSE_TYPES:
                record.outcome = AggregationOutcome.skipped_unavailable
                return None
            reference = get_version_weights(session, latest.id)
            if reference is None:
                record.outcome = AggregationOutcome.skipped_unavailable
                return None
            record.base_weights_id = reference.id

            cap = cohort_cap(latest.weight_count, FED_AGG_MEMORY_BYTES)
            newest_per_user = (
                select(ClientDeltaSubmission.id, ClientDeltaSubmission.created_at)
                .where(ClientDeltaSubmission.base_weights_id == reference.id,
                       ClientDeltaSubmission.valid == True)  # noqa: E712
                .distinct(ClientDeltaSubmission.user_id)
                .order_by(ClientDeltaSubmission.user_id,
                          ClientDeltaSubmission.created_at.desc())  # type: ignore[attr-defined]
                .subquery())
            ids = list(session.execute(
                select(newest_per_user.c.id)
                .order_by(newest_per_user.c.created_at.desc())
                .limit(cap)).scalars())
            record.cap = cap
            record.cohort = len(ids)
            record.users = session.execute(
                select(func.count()).select_from(newest_per_user)).scalar_one()
            record.on_base = session.execute(
                select(func.count()).select_from(ClientDeltaSubmission)
                .where(ClientDeltaSubmission.base_weights_id == reference.id)).scalar_one()

        if len(ids) < FED_MIN_SUBMISSIONS:
            record.outcome = AggregationOutcome.skipped_min_submissions
            return None

        with timer(AggregationStage.blob_fetch):
            deltas = np.empty((len(ids), latest.weight_count), dtype=np.float32)
            rows = session.execute(
                select(ClientDeltaSubmission.deltas)
                .where(ClientDeltaSubmission.id.in_(ids))  # type: ignore[attr-defined]
                .execution_options(yield_per=64)).scalars()
            filled = 0
            for blob in rows:
                deltas[filled] = np.frombuffer(blob, dtype=np.float32)
                filled += 1
            deltas = deltas[:filled]
            weights = np.frombuffer(decompress(reference.weights), dtype=np.float32)
        record.cohort = filled

        return RoundPlan(latest.id, latest.fingerprint, latest.contract_version,
                         reference.id, weights, deltas)


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
        latest = (select(ModelVersion.model_key, ModelVersion.submission_type)
                  .distinct(ModelVersion.model_key)
                  .order_by(ModelVersion.model_key,
                            ModelVersion.version.desc())  # type: ignore[attr-defined]
                  .subquery())
        keys = list(session.execute(
            select(latest.c.model_key)
            .where(latest.c.submission_type.in_(DENSE_TYPES))).scalars())
    for key in keys:
        federated_aggregation.delay(key)
    return keys
