from dataclasses import dataclass

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
from worker.baking import bake, store
from worker.celery_app import app
from worker.compute import cohort_cap, dense_update
from worker.locking import model_lock
from worker.phases import Phases

log = get_task_logger(__name__)

DENSE_TYPES = (SubmissionType.raw, SubmissionType.quantize)


@dataclass(frozen=True)
class RoundPlan:
    version_id: int
    fingerprint: str
    contract_version: int
    reference_id: int
    reference: np.ndarray
    deltas: np.ndarray
    on_base: int
    detail: str


@dataclass(frozen=True)
class Outcome:
    outcome: str
    detail: str
    cohort: int = 0
    on_base: int = 0


def _read_round(key: str, phases: Phases) -> RoundPlan | Outcome:
    with Session(engine) as session:
        with phases("submission_metadata"):
            latest = get_latest_version(session, key)
            if latest is None:
                return Outcome("skipped_unavailable", "no seeded version")
            if latest.submission_type not in DENSE_TYPES:
                return Outcome("skipped_unavailable",
                            f"no dense aggregation for '{latest.submission_type.value}'")
            reference = get_version_weights(session, latest.id)
            if reference is None:
                return Outcome("skipped_unavailable", "no active global weights")

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
            users = session.execute(select(func.count()).select_from(newest_per_user)).scalar_one()
            on_base = session.execute(
                select(func.count()).select_from(ClientDeltaSubmission)
                .where(ClientDeltaSubmission.base_weights_id == reference.id)).scalar_one()

        detail = f"{len(ids)} of {users} users, {on_base} submissions on this base, cap {cap}"
        if cap < FED_MIN_SUBMISSIONS or (FED_TRIM_RATIO and cap < 1 / FED_TRIM_RATIO):
            detail += " (memory cap below FED_MIN_SUBMISSIONS or 1/FED_TRIM_RATIO)"
        if len(ids) < FED_MIN_SUBMISSIONS:
            return Outcome("skipped_min_submissions",
                        f"{detail}; min {FED_MIN_SUBMISSIONS}", len(ids), on_base)

        with phases("blob_fetch"):
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

        return RoundPlan(latest.id, latest.fingerprint, latest.contract_version,
                         reference.id, weights, deltas, on_base, detail)


def _aggregate(key: str, phases: Phases) -> Outcome:
    plan = _read_round(key, phases)
    if isinstance(plan, Outcome):
        return plan
    cohort = len(plan.deltas)

    with phases("runtime"):
        rt = runtime.get(key)
    if rt.fingerprint != plan.fingerprint:
        return Outcome("skipped_unavailable", "no seeded version matching the running code",
                    cohort, plan.on_base)

    with phases("trimmed_mean"):
        new_weights = dense_update(plan.reference, plan.deltas, FED_TRIM_RATIO)
    try:
        baked = bake(rt, new_weights, plan.contract_version, phases)
    except Exception as exc:
        return Outcome("export_failed", f"{plan.detail}; {exc}", cohort, plan.on_base)

    try:
        with phases("commit"), Session(engine) as session:
            store(session, key, plan.version_id, plan.reference_id, baked)
            session.commit()
    except IntegrityError:
        return Outcome("duplicate_round", f"base {plan.reference_id} already aggregated",
                    cohort, plan.on_base)

    with phases("clear_model_limits"):
        clear_model_limits(key)
    return Outcome("aggregated", plan.detail, cohort, plan.on_base)


@app.task(name=FED_AGG_TASK)
def federated_aggregation(model_key: str) -> dict:
    phases = Phases(task="aggregation", model_key=model_key)
    with model_lock(f"agg:{model_key}", FED_LOCK_TTL_SECONDS) as held:
        if not held:
            outcome = Outcome("skipped_locked", "another round holds the lock")
        else:
            try:
                outcome = _aggregate(model_key, phases)
            except Exception as exc:
                outcome = Outcome("failed", str(exc))
    log.info("%s: %s (%s)", model_key, outcome.outcome, outcome.detail)
    return {"model_key": model_key, "outcome": outcome.outcome, "cohort": outcome.cohort,
            "on_base": outcome.on_base, "detail": outcome.detail,
            "timings": phases.timings}


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
