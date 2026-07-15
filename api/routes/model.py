import base64
import uuid
from datetime import datetime

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from worker.celery_app import app as celery_app

from common.celery_tasks import QUANTIZE_TASK, VALIDATE_TASK
from common.config import (
    DOWNLOAD_COOLDOWN_SECONDS,
    QUANTIZE_DAILY_LIMIT,
    QUANTIZE_DAILY_WINDOW_SECONDS,
    RESULT_POLL_TIMEOUT_SECONDS,
    SUBMIT_DAILY_LIMIT,
    SUBMIT_DAILY_WINDOW_SECONDS,
)
from common.db import (
    Artifact,
    GlobalWeights,
    JobStatus,
    ModelDefinition,
    ModelVersion,
    QuantizationJob,
    QuantizationResult,
    SubmissionType,
    User,
    ClientDeltaSubmission,
    engine,
    get_latest_version,
    get_latest_weights,
    get_model_def,
    get_session,
    get_version_weights,
    get_weights_artifact,
    list_model_defs,
    utcnow,
)
from common.ratelimit import RateLimit
from api.lib.ratelimit import check_limit, record_usage
from api.lib.session import get_current_user
from api.lib.challenge import require_device_owner

router = APIRouter(prefix="/model")


FINGERPRINT_HEADER = "X-Model-Fingerprint"
MODEL_VERSION_HEADER = "X-Model-Version"
WEIGHTS_ID_HEADER = "X-Weights-ID"
WEIGHTS_TIMESTAMP_HEADER = "X-Weights-Timestamp"
SIGNATURE_HEADER = "X-Model-Signature"
CONTRACT_VERSION_HEADER = "X-Contract-Version"
NORM_PARAMS_HEADER = "X-Norm-Params"


class ModelInfo(BaseModel):
    key: str
    name: str
    firmware_id: int | None
    version: int
    min_app_version: str
    fingerprint: str
    contract_version: int
    submission_type: SubmissionType
    weight_count: int
    weights_version: datetime | None


class ModelVersionInfo(BaseModel):
    version: int
    fingerprint: str
    contract_version: int
    submission_type: SubmissionType
    min_app_version: str
    weight_count: int
    created_at: datetime
    weights_version: datetime | None


def require_model(session: Session, key: str) -> ModelDefinition:
    meta = get_model_def(session, key)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found")
    return meta


def require_submission_type(session: Session, key: str,
                            accepted: set[SubmissionType]) -> None:
    latest = get_latest_version(session, key)
    if latest is None or latest.submission_type not in accepted:
        raise HTTPException(status_code=404, detail="Not found")


def check_submission(session: Session, key: str, weights_id: int,
                     body: bytes) -> GlobalWeights:
    require_model(session, key)

    base = session.get(GlobalWeights, weights_id)
    if base is None or base.model_key != key:
        raise HTTPException(status_code=400,
                            detail=f"Unknown base weights for model '{key}'")

    active = get_latest_weights(session, key)
    if active is None or base.id != active.id:
        raise HTTPException(
            status_code=409,
            detail=f"Stale base weights; re-download the latest weights of "
                   f"'{key}' before submitting")

    if len(body) != base.weight_count * 4:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {base.weight_count} little-endian float32 weights")
    if not np.all(np.isfinite(np.frombuffer(body, dtype=np.float32))):
        raise HTTPException(status_code=400,
                            detail="Weights contain non-finite values")
    return base


def store_submission(session: Session, base: GlobalWeights, body: bytes,
                     user: User) -> ClientDeltaSubmission:
    submission = ClientDeltaSubmission(
        user_id=user.id,
        model_key=base.model_key,
        base_weights_id=base.id,
        version_id=base.version_id,
        deltas=bytes(body),
        weight_count=base.weight_count,
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)
    return submission


def _version_info(session: Session, version: ModelVersion) -> ModelVersionInfo:
    weights = get_version_weights(session, version.id)
    return ModelVersionInfo(
        version=version.version, fingerprint=version.fingerprint,
        contract_version=version.contract_version,
        submission_type=version.submission_type,
        min_app_version=version.min_app_version,
        weight_count=version.weight_count, created_at=version.created_at,
        weights_version=weights.created_at if weights else None,
    )


@router.get("/list", response_model=list[ModelInfo])
def list_models(session: Session = Depends(get_session),
                user: User = Depends(get_current_user)):
    out = []
    for meta in list_model_defs(session):
        latest = get_latest_version(session, meta.key)
        if latest is None:
            continue
        weights = get_version_weights(session, latest.id)
        out.append(ModelInfo(
            key=meta.key, name=meta.name,
            firmware_id=meta.firmware_id,
            version=latest.version, min_app_version=latest.min_app_version,
            fingerprint=latest.fingerprint,
            contract_version=latest.contract_version,
            submission_type=latest.submission_type,
            weight_count=latest.weight_count,
            weights_version=weights.created_at if weights else None,
        ))
    return out


@router.get("/versions/{key}", response_model=list[ModelVersionInfo])
def list_versions(key: str,
                  session: Session = Depends(get_session),
                  user: User = Depends(get_current_user)):
    require_model(session, key)
    versions = session.exec(
        select(ModelVersion)
        .where(ModelVersion.model_key == key)
        .order_by(ModelVersion.version.desc())  # type: ignore[attr-defined]
    ).all()
    return [_version_info(session, v) for v in versions]


@router.post("/submit/quantize/{key}/{weights_id}", status_code=202)
async def quantize_model(key: str, weights_id: int, request: Request,
                         session: Session = Depends(get_session),
                         user: User = Depends(get_current_user)):
    check_limit(RateLimit.weight_submit, user.id, key,
                QUANTIZE_DAILY_LIMIT, QUANTIZE_DAILY_WINDOW_SECONDS)
    require_submission_type(session, key, {SubmissionType.quantize})
    require_device_owner(session, user)
    body = await request.body()
    base = check_submission(session, key, weights_id, body)

    try:
        submission = store_submission(session, base, body, user)
        job = QuantizationJob(submission_id=submission.id, model_key=key)
        session.add(job)
        session.commit()
        session.refresh(job)
        # The job id doubles as the task id so the result endpoint can wait on it.
        celery_app.send_task(QUANTIZE_TASK, args=[str(job.id)], task_id=str(job.id))
        return {"job_id": str(job.id)}
    finally:
        record_usage(RateLimit.weight_submit, user.id, key, QUANTIZE_DAILY_WINDOW_SECONDS)


@router.post("/submit/raw/{key}/{weights_id}", status_code=202)
async def submit_weights(key: str, weights_id: int, request: Request,
                         session: Session = Depends(get_session),
                         user: User = Depends(get_current_user)):
    check_limit(RateLimit.weight_submit, user.id, key,
                SUBMIT_DAILY_LIMIT, SUBMIT_DAILY_WINDOW_SECONDS)
    require_submission_type(session, key,
                            {SubmissionType.raw, SubmissionType.quantize})
    require_device_owner(session, user)
    body = await request.body()
    base = check_submission(session, key, weights_id, body)

    try:
        submission = store_submission(session, base, body, user)
        celery_app.send_task(VALIDATE_TASK, args=[submission.id])
        return {"submission_id": submission.id}
    finally:
        record_usage(RateLimit.weight_submit, user.id, key, SUBMIT_DAILY_WINDOW_SECONDS)


def _settled_result(session: Session, job_id: uuid.UUID, user: User) -> Response | None:
    job = session.get(QuantizationJob, job_id)
    if job is None or job.status == JobStatus.expired:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    submission = session.get(ClientDeltaSubmission, job.submission_id)
    if submission is None or submission.user_id != user.id:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    if job.status in (JobStatus.pending, JobStatus.running):
        return None

    if job.status == JobStatus.failed:
        return JSONResponse(status_code=422,
                            content={"status": job.status.value, "error": job.error})

    # A done job whose result the sweep already reaped reads as expired, not as
    # a server error (the status flips in the same transaction as the delete, so
    # this only races a concurrent sweep).
    result = session.get(QuantizationResult, job.id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    job.served_at = utcnow()
    session.add(job)
    session.commit()

    version = session.get(ModelVersion, submission.version_id)
    headers = {
        CONTRACT_VERSION_HEADER: str(version.contract_version),
        NORM_PARAMS_HEADER: base64.b64encode(version.norm_params).decode(),
        "Content-Disposition": f'attachment; filename="{job.model_key}.tflite"',
    }
    if job.signature is not None:
        headers[SIGNATURE_HEADER] = base64.b64encode(job.signature).decode()

    return Response(content=result.data, media_type="application/octet-stream",
                    headers=headers)


@router.get("/quantize/result/{job_id}")
def quantize_result(job_id: uuid.UUID, user: User = Depends(get_current_user)):
    with Session(engine) as session:
        settled = _settled_result(session, job_id, user)
        if settled is not None:
            return settled

    try:
        celery_app.AsyncResult(str(job_id)).get(
            timeout=RESULT_POLL_TIMEOUT_SECONDS, propagate=False)
    except Exception:
        pass  # timeout (or a backend hiccup) — report whatever state the DB holds

    with Session(engine) as session:
        settled = _settled_result(session, job_id, user)
        if settled is not None:
            return settled
        job = session.get(QuantizationJob, job_id)
        return JSONResponse(status_code=202, content={"status": job.status.value})


@router.get("/download/{artifact}/{key}")
def download_model(artifact: Artifact, key: str, version: int | None = None,
                   session: Session = Depends(get_session),
                   user: User = Depends(get_current_user)):
    check_limit(RateLimit.model_download, user.id, key, 1, DOWNLOAD_COOLDOWN_SECONDS)
    require_device_owner(session, user)
    require_model(session, key)

    if version is None:
        ver = get_latest_version(session, key)
    else:
        ver = session.exec(
            select(ModelVersion).where(ModelVersion.model_key == key,
                                       ModelVersion.version == version)
        ).first()
    if ver is None:
        raise HTTPException(status_code=404, detail=f"No such version of '{key}'")

    weights = get_version_weights(session, ver.id)
    if weights is None:
        raise HTTPException(status_code=404, detail=f"No weights for model '{key}'")

    baked = get_weights_artifact(session, weights.id, artifact)
    if baked is None:
        raise HTTPException(status_code=404,
                            detail=f"No {artifact.value} artifact for model '{key}'")

    # The cooldown is spent only once we actually serve the artifact, so the
    # cheap rejections above (unknown version/weights/artifact) don't count.
    try:
        headers = {
            FINGERPRINT_HEADER: ver.fingerprint,
            MODEL_VERSION_HEADER: str(ver.version),
            WEIGHTS_ID_HEADER: str(weights.id),
            WEIGHTS_TIMESTAMP_HEADER: weights.created_at.isoformat(),
            "Content-Disposition":
                f'attachment; filename="{key}-v{ver.version}-{artifact.value}.tflite"',
        }
        if artifact is Artifact.quantized:
            headers[CONTRACT_VERSION_HEADER] = str(ver.contract_version)
            headers[NORM_PARAMS_HEADER] = base64.b64encode(ver.norm_params).decode()
            if weights.artifact_signature is not None:
                headers[SIGNATURE_HEADER] = base64.b64encode(weights.artifact_signature).decode()

        # zstd-compressed as stored; the client decompresses.
        return Response(content=baked.data, media_type="application/octet-stream",
                        headers=headers)
    finally:
        record_usage(RateLimit.model_download, user.id, key, DOWNLOAD_COOLDOWN_SECONDS)
