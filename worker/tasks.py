import uuid
from datetime import timedelta, datetime

import numpy as np
import tensorflow as tf
from celery.signals import worker_process_init
from sqlalchemy import and_, delete, or_
from sqlmodel import Session, select

from worker.celery_app import app
from worker.utils.weight_validation import (
    compute_mse_threshold,
    filter_outliers,
    malformed_reason,
    update_magnitude,
    validate_submission,
)

from common.celery_tasks import (
    CLEANUP_TASK,
    FED_AGG_TASK,
    QUANTIZE_TASK,
    SECURE_AGG_TASK,
    VALIDATE_TASK,
)
from common.config import (
    DATASETS_DIR,
    FED_MIN_SUBMISSIONS,
    RESULT_TTL_SECONDS,
    SERVE_GRACE_SECONDS,
    SERVER_PRIVATE_KEY_FILE,
)
from common.db import (
    Artifact,
    GlobalWeights,
    JobStatus,
    QuantizationJob,
    QuantizationResult,
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    SubmissionType,
    ClientDeltaSubmission,
    WeightsArtifact,
    engine,
    get_latest_version,
    get_latest_weights,
    get_version_weights,
    utcnow,
)

from common.compression import compress
from common.ratelimit import clear_model_limits
from common.secure_agg import dequantize, ring_sum

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

@worker_process_init.connect
def _load_models(**_) -> None:
    for key in MODELS:
        try:
            trainer = MODELS[key].build_trainer(DATASETS_DIR)
            fingerprint = trainer.arch_fingerprint()
            rep = trainer.representative_dataset(data_root=DATASETS_DIR)
            _models[key] = (trainer.model, rep, fingerprint,
                            trainer.contract_version, trainer.norm_param_bytes())
        except Exception as exc:  # missing dataset / build error — skip, don't crash boot
            print(f"[worker] model '{key}' unavailable, skipping: {exc}")


@app.task(name=QUANTIZE_TASK)
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

        submission = session.get(ClientDeltaSubmission, job.submission_id)
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

            reason = malformed_reason(submission, model.total_weight_size)
            if reason is not None:
                submission.valid = False
                session.add(submission)
                raise ValueError(f"invalid submission: {reason}")

            # Aggregation-usability is judged silently: the verdict is cached on
            # the row and the artifact is produced either way.
            submission.valid = validate_submission(
                submission, model.total_weight_size,
                get_latest_weights(session, job.model_key)) is None
            session.add(submission)

            # The personalized int8 artifact is built from the client's own local
            # weights, reconstructed as base snapshot + submitted delta.
            base = session.get(GlobalWeights, submission.base_weights_id)
            if base is None:
                raise ValueError(f"base weights {submission.base_weights_id} not found")
            delta = np.frombuffer(submission.deltas, dtype=np.float32)
            local = (np.frombuffer(base.weights, dtype=np.float32) + delta).astype(np.float32)
            model.restore(tf.constant(local, dtype=tf.float32))
            optimized = bytes(get_optimized_model(model, rep_dataset))
            job.signature = sign_model(optimized, contract_version, norm_bytes,
                                       SERVER_PRIVATE_KEY_FILE)
            session.add(QuantizationResult(job_id=job.id, data=compress(optimized)))
            job.status = JobStatus.done
        except Exception as exc:  # surfaced to the client via the result endpoint
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utcnow()
            session.add(job)
            session.commit()


@app.task(name=VALIDATE_TASK)
def validate_weight_submission(submission_id: int) -> None:
    """Background verdict for a submit-only upload. Cached on the row for
    aggregation; never surfaced to the client."""
    with Session(engine) as session:
        submission = session.get(ClientDeltaSubmission, submission_id)
        if submission is None or submission.model_key not in _models:
            return
        model, _, _, _, _ = _models[submission.model_key]
        reason = validate_submission(submission, model.total_weight_size,
                                     get_latest_weights(session, submission.model_key))
        submission.valid = reason is None
        if reason is not None:
            print(f"[validate] {submission.model_key}: "
                  f"submission {submission.id} rejected: {reason}")
        session.add(submission)
        session.commit()


def _bake_and_store(session: Session, key: str, latest, new_weights: np.ndarray,
                    mse_threshold: float | None, model, rep_dataset,
                    contract_version: int, norm_bytes: bytes) -> Exception | None:
    model.restore(tf.constant(new_weights, dtype=tf.float32))
    trainable = quantized = signature = export_error = None
    try:
        trainable = bytes(get_trainable_model(model))
        quantized = bytes(get_optimized_model(model, rep_dataset))
        signature = sign_model(quantized, contract_version, norm_bytes,
                               SERVER_PRIVATE_KEY_FILE)
    except Exception as exc:
        export_error = exc

    snapshot = GlobalWeights(
        model_key=key, version_id=latest.id,
        weights=new_weights.tobytes(),
        weight_count=model.total_weight_size,
        valid=export_error is None,
        mse_threshold=mse_threshold,
        artifact_signature=signature,
    )
    session.add(snapshot)
    if export_error is None:
        # Artifacts commit with the snapshot they were baked from, so a visible
        # row always has them (signature covers the raw bytes — compress after).
        session.flush()  # for the row id the artifacts are keyed by
        session.add(WeightsArtifact(weights_id=snapshot.id,
                                    artifact=Artifact.trainable,
                                    data=compress(trainable)))
        session.add(WeightsArtifact(weights_id=snapshot.id,
                                    artifact=Artifact.quantized,
                                    data=compress(quantized)))
    session.commit()
    if export_error is None:
        clear_model_limits(key)
    return export_error


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
        select(ClientDeltaSubmission)
        .where(ClientDeltaSubmission.version_id == latest.id,
               ClientDeltaSubmission.created_at > cutoff)
        .order_by(ClientDeltaSubmission.created_at.asc())  # type: ignore
    ))
    latest_per_user = {sub.user_id: sub for sub in submissions}

    valid: list[ClientDeltaSubmission] = []
    for sub in latest_per_user.values():
        if sub.valid is None:  # never validated by the quantize/submit tasks
            reason = validate_submission(sub, model.total_weight_size, reference)
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

    deltas = np.stack([np.frombuffer(sub.deltas, dtype=np.float32)
                       for sub in valid])
    kept = deltas[filter_outliers(deltas)]

    # FedAvg over deltas: new global = reference global + mean of the accepted
    # updates. With a shared base this is identical to averaging absolute weights.
    reference_weights = np.frombuffer(reference.weights, dtype=np.float32)
    new_weights = (reference_weights + fed_avg(kept)).astype(np.float32)

    # Bake the new weights into fresh serving artifacts. A failed export
    # invalidates the round: clients keep pulling the previous snapshot, and the
    # window's submissions stay consumed.
    export_error = _bake_and_store(
        session, key, latest, new_weights,
        compute_mse_threshold([update_magnitude(delta) for delta in kept]),
        model, rep_dataset, contract_version, norm_bytes)
    if export_error is not None:
        return (f"aggregated {len(kept)} submissions but artifact export failed "
                f"(round invalidated): {export_error}")

    return (f"aggregated {len(kept)} submissions "
            f"({len(submissions)} in window, {len(latest_per_user)} users, "
            f"{len(valid) - len(kept)} outliers dropped)")


@app.task(name=FED_AGG_TASK)
def federated_aggregation(model_key: str | None = None) -> dict[str, str]:
    keys = [model_key] if model_key is not None else list(_models)
    summary: dict[str, str] = {}
    with Session(engine) as session:
        for key in keys:
            summary[key] = _aggregate_model(session, key)
            print(f"[aggregation] {key}: {summary[key]}")
    return summary


def _fail_round(session: Session, round: SecureRound, reason: str) -> str:
    round.status = SecureRoundStatus.failed
    round.finished_at = utcnow()
    session.add(round)
    session.commit()
    return f"failed: {reason}"


@app.task(name=SECURE_AGG_TASK)
def secure_aggregation(round_id: int) -> str:
    with Session(engine) as session:
        round = session.get(SecureRound, round_id)
        if round is None:
            return "skipped: round not found"
        if round.status != SecureRoundStatus.sealed:
            return f"skipped: round is {round.status.value}, not sealed"

        key = round.model_key
        if key not in _models:
            return "skipped: model not initialized"
        model, rep_dataset, fingerprint, contract_version, norm_bytes = _models[key]

        latest = get_latest_version(session, key)
        if latest is None or latest.id != round.version_id \
                or latest.fingerprint != fingerprint:
            return _fail_round(session, round, "round version is no longer current")

        members = list(session.exec(
            select(SecureRoundMember)
            .where(SecureRoundMember.round_id == round_id)))
        submitted = [m for m in members if m.masked is not None]
        if len(submitted) != len(members) or len(members) != round.member_count:
            return _fail_round(
                session, round,
                f"{len(submitted)}/{round.member_count} members submitted "
                f"(masks only cancel with the full roster)")

        base = session.get(GlobalWeights, round.base_weights_id)
        if base is None:
            return _fail_round(session, round, "base weights missing")

        expected = model.total_weight_size
        vectors = []
        for m in members:
            v = np.frombuffer(m.masked, dtype="<u4").astype(np.uint32)
            if v.size != expected:
                return _fail_round(session, round,
                                   f"member {m.user_id} vector length mismatch")
            vectors.append(v)

        mean_delta = dequantize(ring_sum(vectors), round.scale, round.member_count)
        reference = np.frombuffer(base.weights, dtype=np.float32)
        new_weights = (reference + mean_delta).astype(np.float32)

        # No individual update is visible, so the only guard is aggregate-level:
        # the mean of clipped deltas must be finite and stay within the clip bound.
        if not np.all(np.isfinite(new_weights)) \
                or float(np.max(np.abs(mean_delta))) > round.clip_bound * 1.001:
            return _fail_round(session, round,
                               "aggregate failed sanity check (implausible mean delta)")

        export_error = _bake_and_store(session, key, latest, new_weights, None,
                                       model, rep_dataset, contract_version, norm_bytes)
        round.status = (SecureRoundStatus.failed if export_error is not None
                        else SecureRoundStatus.aggregated)
        round.finished_at = utcnow()
        session.add(round)
        session.commit()
        if export_error is not None:
            return (f"aggregated {len(members)} members but artifact export failed "
                    f"(round invalidated): {export_error}")
        return f"aggregated {len(members)} members into new global weights"


@app.task(name=CLEANUP_TASK)
def cleanup_results() -> int:
    """Drop results for jobs that were served (after a grace window) or never
    claimed (after the TTL): the result row is deleted and the signature nulled.
    Weight submissions are left untouched. The join keeps the sweep off the
    blobs themselves — it only ever reads job columns."""
    now = utcnow()
    grace_cutoff = now - timedelta(seconds=SERVE_GRACE_SECONDS)
    ttl_cutoff = now - timedelta(seconds=RESULT_TTL_SECONDS)
    cleaned = 0
    with Session(engine) as session:
        stmt = select(QuantizationJob).join(
            QuantizationResult,
            QuantizationResult.job_id == QuantizationJob.id,  # type: ignore
        ).where(
            or_(
                and_(
                    QuantizationJob.served_at.is_not(None),  # type: ignore
                    QuantizationJob.served_at < grace_cutoff,
                ),
                QuantizationJob.created_at < ttl_cutoff,
            ),
        )
        jobs = list(session.exec(stmt))
        if jobs:
            session.execute(delete(QuantizationResult).where(
                QuantizationResult.job_id.in_([job.id for job in jobs])))  # type: ignore
            for job in jobs:
                job.signature = None
                job.status = JobStatus.expired
                session.add(job)
            cleaned = len(jobs)
        session.commit()
    return cleaned
