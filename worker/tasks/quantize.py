import uuid
from dataclasses import dataclass

import numpy as np
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
from worker.baking import bake_quantized
from worker.celery_app import app
from worker.phases import Phases


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


def _fail(job_id: uuid.UUID, error: str) -> None:
    with Session(engine) as session:
        QuantizationJob.transition(session, job_id, JobStatus.running, JobStatus.failed,
                                   error=error, finished_at=utcnow())
        session.commit()


@app.task(name=QUANTIZE_TASK)
def quantize_submission(job_id: str) -> str:
    pk = uuid.UUID(job_id)
    if not QuantizationJob.claim(pk, JobStatus.pending, JobStatus.running,
                                 started_at=utcnow()):
        return "skipped: job already claimed"

    phases = Phases(task="quantize", job_id=job_id)
    try:
        with phases("read"):
            plan = _read(pk)
        with phases("runtime"):
            rt = runtime.get(plan.model_key)
        if rt.fingerprint != plan.fingerprint:
            raise ValueError(f"stale model version for '{plan.model_key}'")

        local = (plan.reference + plan.delta).astype(np.float32)
        data, signature = bake_quantized(rt, local, plan.contract_version, phases)

        with phases("commit"), Session(engine) as session:
            session.add(QuantizationResult(job_id=pk, data=data))
            if not QuantizationJob.transition(session, pk, JobStatus.running, JobStatus.done,
                                              signature=signature, finished_at=utcnow()):
                session.rollback()
                return "skipped: job no longer running"
            session.commit()
        return "done"
    except Exception as exc:
        _fail(pk, str(exc))
        return f"failed: {exc}"
