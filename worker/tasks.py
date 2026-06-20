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

from ml.models.feature_mlp import get_trainer as get_feature_mlp_trainer
from ml.saving import get_optimized_model

# model_key -> Trainer builder (TensorFlow). The DB ModelDefinition registry is
# the source of truth for which models exist; this just attaches the builder the
# worker uses to materialize each one.
_TRAINER_BUILDERS = {
    "feature-mlp": get_feature_mlp_trainer,
}

# Per-process cache of (model, representative_dataset) so TensorFlow and the
# calibration data load once per worker, not once per job.
_cache: dict[str, tuple] = {}


def _model_and_rep(model_key: str):
    if model_key not in _TRAINER_BUILDERS:
        raise ValueError(f"unknown model '{model_key}'")
    if model_key not in _cache:
        trainer = _TRAINER_BUILDERS[model_key](DATASETS_DIR, SEED)
        eval_ds = trainer.combine(trainer.subject_datasets(DATASETS_DIR, SEED)[1])
        _cache[model_key] = (trainer.model, trainer.representative_dataset(eval_ds))
    return _cache[model_key]


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
            model, rep_dataset = _model_and_rep(job.model_key)
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
