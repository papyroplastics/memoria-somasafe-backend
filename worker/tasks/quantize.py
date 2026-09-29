import uuid
from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from celery.utils.log import get_task_logger
from sqlmodel import Session

from common.celery_tasks import QUANTIZE_TASK
from common.compression import decompress
from common.db import (
    ClientDeltaSubmission,
    GlobalWeights,
    JobStatus,
    QuantizationJob,
    QuantizationResult,
    engine,
    get_latest_version,
    utcnow,
)
from worker import runtime
from worker.baking import QuantizedBake, Restore, bake_quantized, restore
from worker.celery_app import app
from worker.metrics import Timer, write

log = get_task_logger(__name__)


class QuantizeStage(StrEnum):
    read = auto()
    runtime = auto()
    apply_delta = auto()
    commit = auto()


STAGES = (QuantizeStage.read, QuantizeStage.runtime, QuantizeStage.apply_delta,
          *Restore, *QuantizedBake, QuantizeStage.commit)


class QuantizeOutcome(StrEnum):
    done = auto()
    skipped_claimed = auto()
    skipped_not_running = auto()
    failed = auto()


@dataclass
class QuantizeRecord:
    job_id: str
    outcome: QuantizeOutcome | None = None
    model_key: str | None = None


@dataclass(frozen=True)
class QuantizePlan:
    model_key: str
    fingerprint: str
    contract_version: int
    reference: np.ndarray
    delta: np.ndarray


def _read(job_id: uuid.UUID) -> QuantizePlan:
    with Session(engine) as session:
        job = session.get(QuantizationJob, job_id)
        submission = session.get(ClientDeltaSubmission, job.submission_id)
        if submission is None:
            raise ValueError(f"submission {job.submission_id} not found")
        base = session.get(GlobalWeights, submission.base_weights_id)
        if base is None:
            raise ValueError(f"base weights {submission.base_weights_id} not found")
        latest = get_latest_version(session, job.model_key)
        if latest is None or base.version_id != latest.id:
            raise ValueError(f"stale model version for '{job.model_key}'")
        return QuantizePlan(
            model_key=job.model_key, fingerprint=latest.fingerprint,
            contract_version=latest.contract_version,
            reference=np.frombuffer(decompress(base.weights), dtype=np.float32),
            delta=np.frombuffer(submission.deltas, dtype=np.float32),
        )


def _fail(job_id: uuid.UUID) -> None:
    with Session(engine) as session:
        QuantizationJob.transition(session, job_id, JobStatus.running, JobStatus.failed,
                                   finished_at=utcnow())
        session.commit()


def _quantize(pk: uuid.UUID, timer: Timer[QuantizeStage | Restore | QuantizedBake],
              record: QuantizeRecord) -> None:
    with timer(QuantizeStage.read):
        plan = _read(pk)
    record.model_key = plan.model_key
    with timer(QuantizeStage.runtime):
        rt = runtime.get(plan.model_key)
    if rt.fingerprint != plan.fingerprint:
        raise ValueError(f"stale model version for '{plan.model_key}'")

    with timer(QuantizeStage.apply_delta):
        local = (plan.reference + plan.delta).astype(np.float32)
    restore(rt.model, local, timer)
    quantized = bake_quantized(rt.model, rt.rep_dataset, plan.contract_version, timer)

    with timer(QuantizeStage.commit), Session(engine) as session:
        session.add(QuantizationResult(job_id=pk, data=quantized.data))
        if not QuantizationJob.transition(session, pk, JobStatus.running, JobStatus.done,
                                          signature=quantized.signature, finished_at=utcnow()):
            session.rollback()
            record.outcome = QuantizeOutcome.skipped_not_running
            return
        session.commit()
    record.outcome = QuantizeOutcome.done


@app.task(name=QUANTIZE_TASK, ignore_result=True)
def quantize_submission(job_id: str) -> None:
    timer = Timer(*STAGES)
    record = QuantizeRecord(job_id)
    pk = uuid.UUID(job_id)
    if not QuantizationJob.claim(pk, JobStatus.pending, JobStatus.running,
                                 started_at=utcnow()):
        record.outcome = QuantizeOutcome.skipped_claimed
    else:
        try:
            _quantize(pk, timer, record)
        except Exception:
            log.exception("quantization job %s failed", job_id)
            record.outcome = QuantizeOutcome.failed
            _fail(pk)
    write(QUANTIZE_TASK, record, timer)
