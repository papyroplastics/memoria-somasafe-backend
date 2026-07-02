import uuid
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sqlalchemy import and_, or_
from sqlmodel import Session, select

from worker.celery_app import app
from worker.utils.weight_validation import (
    compute_mse_threshold,
    filter_outliers,
    mse,
    validate_submission,
)

from common.config import (
    DATASETS_DIR,
    FED_MIN_SUBMISSIONS,
    RESULT_TTL_SECONDS,
    SERVE_GRACE_SECONDS,
)
from common.db import (
    GlobalWeights,
    JobStatus,
    QuantizationJob,
    WeightSubmission,
    engine,
    get_latest_weights,
    utcnow,
)

from ml.model_list import MODELS
from ml.saving import get_optimized_model
from ml.training import fed_avg

# Per-process cache of (model, representative_dataset, fingerprint), built once
# at startup so TensorFlow and the calibration data load once per worker, not
# once per job. Models whose dataset is absent (not trained yet) are skipped —
# the worker only ever quantizes models that scripts.db_seed put in the DB.
_models: dict[str, tuple] = {}


def _init_models() -> None:
    for key in MODELS:
        try:
            trainer = MODELS[key].build_trainer()
            fingerprint = trainer.model.arch_fingerprint()
            rep = trainer.representative_dataset(data_root=DATASETS_DIR)
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
            reason = validate_submission(
                submission, model.total_parameter_size,
                get_latest_weights(session, job.model_key))
            submission.valid = reason is None
            session.add(submission)
            if reason is not None:
                raise ValueError(f"invalid submission: {reason}")
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


def _aggregate_model(session: Session, key: str) -> str:
    if key not in _models:
        return "skipped: model not initialized"
    model, _, fingerprint = _models[key]

    reference = get_latest_weights(session, key)
    if reference is None:
        return "skipped: no active global weights"

    # Window since the newest snapshot regardless of validity: submissions
    # consumed by a later-invalidated round are not re-aggregated.
    cutoff = session.exec(
        select(GlobalWeights.created_at)
        .where(GlobalWeights.model_key == key,
               GlobalWeights.fingerprint == fingerprint)
        .order_by(GlobalWeights.created_at.desc())  # type: ignore[attr-defined]
    ).first()

    submissions = list(session.exec(
        select(WeightSubmission)
        .where(WeightSubmission.model_key == key,
               WeightSubmission.fingerprint == fingerprint,
               WeightSubmission.created_at > cutoff)
        .order_by(WeightSubmission.created_at.asc())  # type: ignore[attr-defined]
    ))
    latest_per_user = {sub.user_id: sub for sub in submissions}

    valid: list[WeightSubmission] = []
    for sub in latest_per_user.values():
        if sub.valid is None:  # never went through quantization's validation
            reason = validate_submission(sub, model.total_parameter_size, reference)
            sub.valid = reason is None
            session.add(sub)
            if reason is not None:
                print(f"[aggregation] {key}: submission {sub.id} rejected: {reason}")
        if sub.valid:
            valid.append(sub)
    session.commit()

    if len(valid) < FED_MIN_SUBMISSIONS:
        return (f"skipped: {len(valid)} valid submissions "
                f"(min {FED_MIN_SUBMISSIONS}, {len(submissions)} in window)")

    vectors = np.stack([np.frombuffer(sub.parameters, dtype=np.float32)
                        for sub in valid])
    kept = vectors[filter_outliers(vectors)]

    reference_params = np.frombuffer(reference.parameters, dtype=np.float32)
    averaged = fed_avg(kept).astype(np.float32)

    session.add(GlobalWeights(
        model_key=key, fingerprint=fingerprint,
        parameters=averaged.tobytes(),
        param_count=model.total_parameter_size,
        mse_threshold=compute_mse_threshold(
            [mse(vector, reference_params) for vector in kept]),
    ))
    session.commit()
    return (f"aggregated {len(kept)} submissions "
            f"({len(submissions)} in window, {len(latest_per_user)} users, "
            f"{len(valid) - len(kept)} outliers dropped)")


@app.task(name="worker.tasks.federated_aggregation")
def federated_aggregation(model_key: str | None = None) -> dict[str, str]:
    """One FedAvg round per model over the submissions accumulated since its
    last GlobalWeights snapshot, keeping only each user's latest. Runs for every
    initialized model unless ``model_key`` narrows it; a model is skipped when
    fewer than FED_MIN_SUBMISSIONS submissions survive validation."""
    keys = [model_key] if model_key is not None else list(_models)
    summary: dict[str, str] = {}
    with Session(engine) as session:
        for key in keys:
            summary[key] = _aggregate_model(session, key)
            print(f"[aggregation] {key}: {summary[key]}")
    return summary


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
