import uuid
from datetime import timedelta, datetime

import numpy as np
import tensorflow as tf
from celery.signals import worker_process_init
from sqlalchemy import and_, or_
from sqlmodel import Session, select

from worker.celery_app import app
from worker.utils.weight_validation import (
    compute_mse_threshold,
    filter_outliers,
    malformed_reason,
    mse,
    validate_submission,
)

from common.config import (
    DATASETS_DIR,
    FED_MIN_SUBMISSIONS,
    RESULT_TTL_SECONDS,
    SERVE_GRACE_SECONDS,
    SERVER_PRIVATE_KEY_FILE,
)
from common.db import (
    GlobalWeights,
    JobStatus,
    QuantizationJob,
    SubmissionType,
    WeightSubmission,
    engine,
    get_latest_version,
    get_latest_weights,
    get_version_weights,
    utcnow,
)

from common.ratelimit import clear_model_limits

from ml.model_list import MODELS
from ml.payload import sign_model
from ml.saving import get_optimized_model, get_trainable_model
from ml.training import fed_avg

# Per-process cache of (model, representative_dataset, fingerprint, contract_version,
# norm_bytes), built once per forked worker child so TensorFlow and the calibration
# data load once per worker, not once per job. Models whose dataset is absent (not
# trained yet) are skipped — the worker only ever touches models that scripts.seed_db
# put in the DB.
#
# Population is deferred to worker_process_init (post-fork): TensorFlow is not
# fork-safe once its runtime and thread pools exist, so initializing it in the
# parent MainProcess would leave every prefork child deadlocked on inherited-locked
# native mutexes the first time it runs a TF op.
_models: dict[str, tuple] = {}


def _init_models() -> None:
    for key in MODELS:
        try:
            trainer = MODELS[key].build_trainer(DATASETS_DIR)
            fingerprint = trainer.arch_fingerprint()
            rep = trainer.representative_dataset(data_root=DATASETS_DIR)
            _models[key] = (trainer.model, rep, fingerprint,
                            trainer.contract_version, trainer.norm_param_bytes())
        except Exception as exc:  # missing dataset / build error — skip, don't crash boot
            print(f"[worker] model '{key}' unavailable, skipping: {exc}")


@worker_process_init.connect
def _load_models(**_) -> None:
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
            model, rep_dataset, fingerprint, contract_version, norm_bytes = _models[job.model_key]

            # Reject weights trained against an outdated version — the worker
            # could not restore them, and aggregation must not mix versions.
            latest = get_latest_version(session, job.model_key)
            if latest is None or submission.version_id != latest.id \
                    or latest.fingerprint != fingerprint:
                raise ValueError(f"stale model version for '{job.model_key}'")

            reason = malformed_reason(submission, model.total_parameter_size)
            if reason is not None:
                submission.valid = False
                session.add(submission)
                raise ValueError(f"invalid submission: {reason}")

            # Aggregation-usability is judged silently: the verdict is cached on
            # the row and the artifact is produced either way.
            submission.valid = validate_submission(
                submission, model.total_parameter_size,
                get_latest_weights(session, job.model_key)) is None
            session.add(submission)

            params = np.frombuffer(submission.parameters, dtype=np.float32).copy()
            model.restore(tf.constant(params, dtype=tf.float32))
            job.result = bytes(get_optimized_model(model, rep_dataset))
            job.signature = sign_model(job.result, contract_version, norm_bytes,
                                       SERVER_PRIVATE_KEY_FILE)
            job.status = JobStatus.done
        except Exception as exc:  # surfaced to the client via the result endpoint
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utcnow()
            session.add(job)
            session.commit()


@app.task(name="worker.tasks.validate_submission")
def validate_weight_submission(submission_id: int) -> None:
    """Background verdict for a submit-only upload. Cached on the row for
    aggregation; never surfaced to the client."""
    with Session(engine) as session:
        submission = session.get(WeightSubmission, submission_id)
        if submission is None or submission.model_key not in _models:
            return
        model, _, _, _, _ = _models[submission.model_key]
        reason = validate_submission(submission, model.total_parameter_size,
                                     get_latest_weights(session, submission.model_key))
        submission.valid = reason is None
        if reason is not None:
            print(f"[validate] {submission.model_key}: "
                  f"submission {submission.id} rejected: {reason}")
        session.add(submission)
        session.commit()


def _aggregate_model(session: Session, key: str) -> str:
    if key not in _models:
        return "skipped: model not initialized"
    model, rep_dataset, fingerprint, contract_version, norm_bytes = _models[key]

    latest = get_latest_version(session, key)
    if latest is None or latest.fingerprint != fingerprint:
        return "skipped: no seeded version matching the running code"

    # Aggregation strategy is chosen by the version's submission type. Today raw
    # and quantize are byte-identical dense vectors and share the FedAvg path
    # below; sparse/DP formats will branch here.
    if latest.submission_type not in (SubmissionType.raw, SubmissionType.quantize):
        return f"skipped: no aggregation strategy for '{latest.submission_type.value}'"

    reference = get_version_weights(session, latest.id)
    if reference is None:
        return "skipped: no active global weights"

    # Window since the newest snapshot regardless of validity: submissions
    # consumed by a later-invalidated round are not re-aggregated.
    cutoff = session.exec(
        select(GlobalWeights.created_at)
        .where(GlobalWeights.version_id == latest.id)
        .order_by(GlobalWeights.created_at.desc())  # type: ignore
    ).first() or datetime.fromtimestamp(0)

    submissions = list(session.exec(
        select(WeightSubmission)
        .where(WeightSubmission.version_id == latest.id,
               WeightSubmission.created_at > cutoff)
        .order_by(WeightSubmission.created_at.asc())  # type: ignore
    ))
    latest_per_user = {sub.user_id: sub for sub in submissions}

    valid: list[WeightSubmission] = []
    for sub in latest_per_user.values():
        if sub.valid is None:  # never validated by the quantize/submit tasks
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

    # Bake the averaged weights into fresh serving artifacts. A failed export
    # invalidates the round: clients keep pulling the previous snapshot, and the
    # window's submissions stay consumed.
    model.restore(tf.constant(averaged, dtype=tf.float32))
    trainable = quantized = signature = export_error = None
    try:
        trainable = bytes(get_trainable_model(model))
        quantized = bytes(get_optimized_model(model, rep_dataset))
        signature = sign_model(quantized, contract_version, norm_bytes,
                               SERVER_PRIVATE_KEY_FILE)
    except Exception as exc:
        export_error = exc

    session.add(GlobalWeights(
        model_key=key, version_id=latest.id,
        parameters=averaged.tobytes(),
        param_count=model.total_parameter_size,
        valid=export_error is None,
        mse_threshold=compute_mse_threshold(
            [mse(vector, reference_params) for vector in kept]),
        trainable_artifact=trainable,
        quantized_artifact=quantized,
        artifact_signature=signature,
    ))
    session.commit()
    if export_error is not None:
        return (f"aggregated {len(kept)} submissions but artifact export failed "
                f"(round invalidated): {export_error}")

    # New weights are live: let every client re-pull and submit again without
    # waiting out the download cooldown / daily submission caps.
    clear_model_limits(key)
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
            QuantizationJob.result.is_not(None),  # type: ignore
            or_(
                and_(
                    QuantizationJob.served_at.is_not(None),  # type: ignore
                    QuantizationJob.served_at < grace_cutoff,
                ),
                QuantizationJob.created_at < ttl_cutoff,
            ),
        )
        for job in session.exec(stmt):
            job.result = None
            job.signature = None
            job.status = JobStatus.expired
            session.add(job)
            cleaned += 1
        session.commit()
    return cleaned
