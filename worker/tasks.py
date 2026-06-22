import uuid
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sqlalchemy import and_, or_
from sqlmodel import Session, select

from worker.celery_app import app

from common.config import DATASETS_DIR, RESULT_TTL_SECONDS, SEED, SERVE_GRACE_SECONDS
from common.db import (
    JobStatus,
    QuantizationJob,
    WeightSubmission,
    engine,
    utcnow,
)

from ml.model_list import MODELS, build_fingerprinted
from ml.saving import get_optimized_model

# Per-process cache of (model, representative_dataset, fingerprint), built once
# at startup so TensorFlow and the calibration data load once per worker, not
# once per job. Models whose dataset is absent (not trained yet) are skipped —
# the worker only ever quantizes models that scripts.db_seed put in the DB.
_models: dict[str, tuple] = {}


def _init_models() -> None:
    for key in MODELS:
        try:
            trainer, fingerprint = build_fingerprinted(key, DATASETS_DIR, SEED)
            eval_ds = trainer.combine(trainer.subject_datasets(DATASETS_DIR, SEED)[1])
            rep = trainer.representative_dataset(eval_ds)
            _models[key] = (trainer.model, rep, fingerprint)
        except Exception as exc:  # missing dataset / build error — skip, don't crash boot
            print(f"[worker] model '{key}' unavailable, skipping: {exc}")


_init_models()


@app.task(name="worker.tasks.quantize_submission")
def quantize_submission(job_id: str) -> None:
    with Session(engine) as session:
        job = session.get(QuantizationJob, uuid.UUID(job_id))
        if job is None:
            return
        job.status = JobStatus.running
        job.started_at = utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)

        submission = session.get(WeightSubmission, job.submission_id)
        try:
            if submission is None:
                raise ValueError(f"submission {job.submission_id} not found")
            if job.model_key not in _models:
                raise ValueError(f"model '{job.model_key}' not initialized")
            model, rep_dataset, fingerprint = _models[job.model_key]
            if submission.fingerprint != fingerprint:
                raise ValueError(
                    f"submission fingerprint {submission.fingerprint} != "
                    f"current {fingerprint} for '{job.model_key}'")
            params = np.frombuffer(submission.parameters, dtype=np.float32).copy()
            model.restore(tf.constant(params, dtype=tf.float32))
            job.result = bytes(get_optimized_model(model, rep_dataset))
            job.status = JobStatus.done
        except Exception as exc:  # surfaced to the client via the result endpoint
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utcnow()
            session.add(job)
            session.commit()


@app.task(name="worker.tasks.cleanup_results")
def cleanup_results() -> int:
    """Drop result bytes for jobs that were served (after a grace window) or
    never claimed (after the TTL). Weight submissions are left untouched."""
    now = utcnow()
    grace_cutoff = now - timedelta(seconds=SERVE_GRACE_SECONDS)
    ttl_cutoff = now - timedelta(seconds=RESULT_TTL_SECONDS)
    cleaned = 0
    with Session(engine) as session:
        stmt = select(QuantizationJob).where(
            QuantizationJob.result.is_not(None),  # type: ignore[union-attr]
            or_(
                and_(
                    QuantizationJob.served_at.is_not(None),  # type: ignore[union-attr]
                    QuantizationJob.served_at < grace_cutoff,
                ),
                QuantizationJob.created_at < ttl_cutoff,
            ),
        )
        for job in session.exec(stmt):
            job.result = None
            job.status = JobStatus.expired
            session.add(job)
            cleaned += 1
        session.commit()
    return cleaned
