import uuid
from datetime import datetime
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlmodel import Session

from worker.celery_app import app as celery_app

from common.config import (
    DOWNLOAD_COOLDOWN_SECONDS,
    QUANTIZE_DAILY_LIMIT,
    QUANTIZE_DAILY_WINDOW_SECONDS,
    MODELS_DIR,
)
from common.db import (
    GlobalWeights,
    JobStatus,
    ModelDefinition,
    ModelFingerprint,
    QuantizationJob,
    User,
    WeightSubmission,
    get_latest_weights,
    get_model_def,
    get_session,
    list_model_defs,
    user_owns_device,
    utcnow,
)
from api.lib.ratelimit import enforce_cooldown, enforce_daily_quota
from .auth import get_current_user

router = APIRouter(prefix="/model")

QUANTIZE_TASK = "worker.tasks.quantize_submission"


FINGERPRINT_HEADER = "X-Model-Fingerprint"
WEIGHTS_ID_HEADER = "X-Weights-ID"
WEIGHTS_TIMESTAMP_HEADER = "X-Weights-Timestamp"


class Artifact(str, Enum):
    trainable = "trainable"
    quantized = "quantized"


class ModelWeights(BaseModel):
    parameters: list[float]
    weights_id: int            # the GlobalWeights snapshot the client trained from


class ModelInfo(BaseModel):
    """What the gateway exposes about a model. The client compares ``fingerprint``
    (architecture) and ``weights_version`` (latest global weights) against what it
    holds locally to decide whether to re-download the model or just its weights."""

    key: str
    name: str
    purpose: str
    firmware_id: int | None
    app_version: str
    fingerprint: str
    version: str               # display_version, human-facing label
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


@router.get("/list", response_model=list[ModelInfo])
def list_models(session: Session = Depends(get_session),
                user: User = Depends(get_current_user)):
    out = []
    for meta in list_model_defs(session):
        fp = session.get(ModelFingerprint, meta.fingerprint)
        latest = get_latest_weights(session, meta.key)
        out.append(ModelInfo(
            key=meta.key, name=meta.name, purpose=meta.purpose.value,
            firmware_id=meta.firmware_id, app_version=meta.app_version,
            fingerprint=meta.fingerprint,
            version=fp.display_version if fp else "",
            weights_version=latest.created_at if latest else None,
        ))
    return out


@router.post("/quantize/{key}", status_code=202)
def quantize_model(key: str, weights: ModelWeights,
                   session: Session = Depends(get_session),
                   user: User = Depends(get_current_user)):
    require_device_owner(session, user)
    enforce_daily_quota("quantize", user.id, key, QUANTIZE_DAILY_LIMIT, QUANTIZE_DAILY_WINDOW_SECONDS)
    meta = require_model(session, key)

    # Resolve the base weights the client trained from; they pin the architecture.
    base = session.get(GlobalWeights, weights.weights_id)
    if base is None or base.model_key != key:
        raise HTTPException(status_code=400,
                            detail=f"Unknown base weights for model '{key}'")

    # Reject weights trained against an outdated architecture — the worker could
    # not restore them, and aggregation must not mix fingerprints.
    if base.fingerprint != meta.fingerprint:
        raise HTTPException(
            status_code=409,
            detail=f"Stale architecture; model '{key}' is now {meta.fingerprint}")

    submission = WeightSubmission(
        user_id=user.id,
        model_key=key,
        base_weights_id=base.id,
        fingerprint=base.fingerprint,
        parameters=np.asarray(weights.parameters, dtype=np.float32).tobytes(),
        param_count=len(weights.parameters),
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)

    job = QuantizationJob(submission_id=submission.id, model_key=key)
    session.add(job)
    session.commit()
    session.refresh(job)

    celery_app.send_task(QUANTIZE_TASK, args=[str(job.id)])

    return {"job_id": str(job.id), "status_url": f"/model/quantize/result/{job.id}"}


@router.get("/quantize/result/{job_id}")
def quantize_result(job_id: uuid.UUID,
                    session: Session = Depends(get_session),
                    user: User = Depends(get_current_user)):
    job = session.get(QuantizationJob, job_id)
    if job is None or job.status == JobStatus.expired:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    # Authorize via the originating submission; 404 (not 403) to keep the id unguessable.
    submission = session.get(WeightSubmission, job.submission_id)
    if submission is None or submission.user_id != user.id:
        raise HTTPException(status_code=404, detail="Result not found or expired")

    if job.status in (JobStatus.pending, JobStatus.running):
        return JSONResponse(status_code=202, content={"status": job.status.value})

    if job.status == JobStatus.failed:
        return JSONResponse(status_code=422,
                            content={"status": job.status.value, "error": job.error})

    # done: stream the int8 .tflite and mark it served (cleanup happens later).
    payload = bytes(job.result)
    job.served_at = utcnow()
    session.add(job)
    session.commit()

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{job.model_key}.tflite"'},
    )


@router.get("/weights/{key}")
def get_weights(key: str,
                session: Session = Depends(get_session),
                user: User = Depends(get_current_user)):
    """Latest global weights (raw float32 buffer) for the model's current
    architecture. Lets a client refresh weights without re-pulling the whole
    model when the fingerprint hasn't changed. Headers carry the architecture,
    the weights id (echoed back to /quantize) and the timestamp (which the
    client compares against to decide when to refresh)."""
    require_device_owner(session, user)
    require_model(session, key)
    weights = get_latest_weights(session, key)
    if weights is None:
        raise HTTPException(status_code=404, detail=f"No weights for model '{key}'")
    return Response(
        content=bytes(weights.parameters),
        media_type="application/octet-stream",
        headers={
            FINGERPRINT_HEADER: weights.fingerprint,
            WEIGHTS_ID_HEADER: str(weights.id),
            WEIGHTS_TIMESTAMP_HEADER: weights.created_at.isoformat(),
            "Content-Disposition": f'attachment; filename="{key}-weights.bin"',
        },
    )


@router.get("/download/{artifact}/{key}", response_class=FileResponse)
def download_model(artifact: Artifact, key: str,
                   session: Session = Depends(get_session),
                   user: User = Depends(get_current_user)):
    """Serve a model's trainable or default int8 artifact from results/<key>/.
    The current architecture fingerprint travels in a header so the client can
    record which architecture the downloaded model belongs to."""
    require_device_owner(session, user)
    enforce_cooldown("download", user.id, key, DOWNLOAD_COOLDOWN_SECONDS)
    meta = require_model(session, key)
    return FileResponse(
        path=MODELS_DIR / key / f"{artifact.value}.tflite",
        filename=f"{key}-{artifact.value}.tflite",
        headers={FINGERPRINT_HEADER: meta.fingerprint},
    )
