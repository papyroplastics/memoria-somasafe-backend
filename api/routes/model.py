import base64
import uuid
from datetime import datetime
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from worker.celery_app import app as celery_app

from common.config import (
    DOWNLOAD_COOLDOWN_SECONDS,
    QUANTIZE_DAILY_LIMIT,
    QUANTIZE_DAILY_WINDOW_SECONDS,
    SUBMIT_DAILY_LIMIT,
    SUBMIT_DAILY_WINDOW_SECONDS,
)
from common.db import (
    GlobalWeights,
    JobStatus,
    ModelDefinition,
    ModelVersion,
    QuantizationJob,
    SubmissionType,
    User,
    ClientDeltaSubmission,
    get_latest_version,
    get_model_def,
    get_session,
    get_version_weights,
    list_model_defs,
    user_owns_device,
    utcnow,
)
from common.storage import weights_artifact_path
from api.lib.ratelimit import enforce_cooldown, enforce_daily_quota
from .auth import get_current_user

router = APIRouter(prefix="/model")

QUANTIZE_TASK = "worker.tasks.quantize_submission"
VALIDATE_TASK = "worker.tasks.validate_submission"


FINGERPRINT_HEADER = "X-Model-Fingerprint"
MODEL_VERSION_HEADER = "X-Model-Version"
WEIGHTS_ID_HEADER = "X-Weights-ID"
WEIGHTS_TIMESTAMP_HEADER = "X-Weights-Timestamp"
SIGNATURE_HEADER = "X-Model-Signature"
CONTRACT_VERSION_HEADER = "X-Contract-Version"
NORM_PARAMS_HEADER = "X-Norm-Params"


class Artifact(str, Enum):
    trainable = "trainable"
    quantized = "quantized"


class ModelInfo(BaseModel):
    """What the gateway exposes about a model's latest version. The client
    compares ``version`` (re-download and reset the local model when it moves —
    the only thing that invalidates a federated epoch) and ``weights_version``
    (re-pull the artifact when it moves) against what it holds locally, and
    checks ``min_app_version`` against its own version before using the model."""

    key: str
    name: str
    purpose: str
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


def require_device_owner(session: Session, user: User) -> None:
    """Gate model access on the user having attested ownership of a device."""
    if not user_owns_device(session, user.id):
        raise HTTPException(status_code=403, detail="No attested device for this user")


def require_submission_type(session: Session, key: str,
                            accepted: set[SubmissionType]) -> None:
    """404 (not 403) when the model's latest version doesn't accept uploads on
    this endpoint. The raw endpoint accepts ``{raw, quantize}`` (quantize's dense
    body is compatible); the quantize endpoint accepts ``{quantize}`` only."""
    latest = get_latest_version(session, key)
    if latest is None or latest.submission_type not in accepted:
        raise HTTPException(status_code=404, detail="Not found")


def store_submission(session: Session, key: str, weights_id: int, body: bytes,
                     user: User) -> ClientDeltaSubmission:
    """Persist a raw-float32 weight *delta* (Δ = local − global, LE float32)
    against the base GlobalWeights snapshot it was computed from. Only
    malformedness is rejected here — whether the update is *usable* for
    aggregation is judged asynchronously and never surfaced to the client."""
    require_model(session, key)

    base = session.get(GlobalWeights, weights_id)
    if base is None or base.model_key != key:
        raise HTTPException(status_code=400,
                            detail=f"Unknown base weights for model '{key}'")

    latest = get_latest_version(session, key)
    if latest is None or base.version_id != latest.id:
        raise HTTPException(
            status_code=409,
            detail=f"Frozen model version; only the latest version of '{key}' "
                   f"accepts submissions")

    if len(body) != base.weight_count * 4:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {base.weight_count} little-endian float32 weights")
    if not np.all(np.isfinite(np.frombuffer(body, dtype=np.float32))):
        raise HTTPException(status_code=400,
                            detail="Weights contain non-finite values")

    submission = ClientDeltaSubmission(
        user_id=user.id,
        model_key=key,
        base_weights_id=base.id,
        version_id=latest.id,
        weights=bytes(body),
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
            key=meta.key, name=meta.name, purpose=meta.purpose.value,
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
    """Every published version of a model, newest first. Only the newest accepts
    submissions; older versions are frozen but still downloadable."""
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
    """Upload locally-trained weights (raw LE float32 body) and get a signed
    int8 artifact built from them; the submission also feeds aggregation."""
    require_submission_type(session, key, {SubmissionType.quantize})
    require_device_owner(session, user)
    enforce_daily_quota("quantize", user.id, key, QUANTIZE_DAILY_LIMIT, QUANTIZE_DAILY_WINDOW_SECONDS)
    submission = store_submission(session, key, weights_id, await request.body(), user)

    job = QuantizationJob(submission_id=submission.id, model_key=key)
    session.add(job)
    session.commit()
    session.refresh(job)

    celery_app.send_task(QUANTIZE_TASK, args=[str(job.id)])

    return {"job_id": str(job.id), "status_url": f"/model/quantize/result/{job.id}"}


@router.post("/submit/raw/{key}/{weights_id}", status_code=202)
async def submit_weights(key: str, weights_id: int, request: Request,
                         session: Session = Depends(get_session),
                         user: User = Depends(get_current_user)):
    """Submit-only federated update (raw LE float32 body): persisted for
    aggregation and validated in the background; nothing comes back."""
    require_submission_type(session, key,
                            {SubmissionType.raw, SubmissionType.quantize})
    require_device_owner(session, user)
    enforce_daily_quota("submit", user.id, key, SUBMIT_DAILY_LIMIT, SUBMIT_DAILY_WINDOW_SECONDS)
    submission = store_submission(session, key, weights_id, await request.body(), user)

    celery_app.send_task(VALIDATE_TASK, args=[submission.id])

    return {"submission_id": submission.id}


@router.get("/quantize/result/{job_id}")
def quantize_result(job_id: uuid.UUID,
                    session: Session = Depends(get_session),
                    user: User = Depends(get_current_user)):
    job = session.get(QuantizationJob, job_id)
    if job is None or job.status == JobStatus.expired:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    # Authorize via the originating submission; 404 (not 403) to keep the id unguessable.
    submission = session.get(ClientDeltaSubmission, job.submission_id)
    if submission is None or submission.user_id != user.id:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    if job.status in (JobStatus.pending, JobStatus.running):
        return JSONResponse(status_code=202, content={"status": job.status.value})

    if job.status == JobStatus.failed:
        return JSONResponse(status_code=422,
                            content={"status": job.status.value, "error": job.error})

    # done: stream the int8 .tflite and mark it served (cleanup happens later).
    # Headers carry the signed fields the app forwards to the ESP32 (ml.payload).
    payload = bytes(job.result)
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

    return Response(content=payload, media_type="application/octet-stream",
                    headers=headers)


@router.get("/download/{artifact}/{key}")
def download_model(artifact: Artifact, key: str, version: int | None = None,
                   session: Session = Depends(get_session),
                   user: User = Depends(get_current_user)):
    """Serve an artifact of the model's newest (or, via ``?version=``, a frozen)
    version, baked with that version's active global weights. The quantized
    artifact additionally carries the signed fields the app forwards to the
    ESP32 (ml.payload)."""
    require_device_owner(session, user)
    enforce_cooldown("download", user.id, key, DOWNLOAD_COOLDOWN_SECONDS)
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

    path = weights_artifact_path(weights.model_key, weights.version_id,
                                 weights.id, artifact.value)
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"No {artifact.value} artifact for model '{key}'")
    blob = path.read_bytes()  # zstd-compressed; the client decompresses

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

    return Response(content=blob, media_type="application/octet-stream",
                    headers=headers)
