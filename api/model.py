import uuid
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
    RESULTS_DIR,
)
from common.db import (
    JobStatus,
    ModelDefinition,
    QuantizationJob,
    User,
    WeightSubmission,
    get_model_def,
    get_session,
    list_model_defs,
    utcnow,
)
from common.ratelimit import enforce_cooldown, enforce_daily_quota
from .auth import get_current_user

router = APIRouter(prefix="/model")

QUANTIZE_TASK = "worker.tasks.quantize_submission"


class Artifact(str, Enum):
    trainable = "trainable"
    quantized = "quantized"


class ModelWeights(BaseModel):
    parameters: list[float]


def require_model(session: Session, key: str, id: int) -> ModelDefinition:
    meta = get_model_def(session, key, id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' version {id} not found")
    return meta


@router.get("/list")
def list_models(session: Session = Depends(get_session),
                user: User = Depends(get_current_user)):
    return list_model_defs(session)


@router.post("/quantize/{key}/{id}", status_code=202)
def quantize_model(key: str, id: int, weights: ModelWeights,
                   session: Session = Depends(get_session),
                   user: User = Depends(get_current_user)):
    require_model(session, key, id)
    enforce_daily_quota("quantize", user.id, key, id,
                        QUANTIZE_DAILY_LIMIT, QUANTIZE_DAILY_WINDOW_SECONDS)

    submission = WeightSubmission(
        model_key=key,
        model_version=id,
        parameters=np.asarray(weights.parameters, dtype=np.float32).tobytes(),
        param_count=len(weights.parameters),
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)

    job = QuantizationJob(submission_id=submission.id, model_key=key, model_version=id)
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


# NOTE: declared last on purpose — the literal /model/list and /model/quantize/*
# routes above must match before this generic {artifact} route, otherwise their
# first path segment would be parsed as an Artifact (and rejected with 422).
@router.get("/{artifact}/{key}/{id}", response_class=FileResponse)
def get_model(artifact: Artifact, key: str, id: int,
              session: Session = Depends(get_session),
              user: User = Depends(get_current_user)):
    """Serve a model's trainable or default int8 artifact from results/<key>/."""
    require_model(session, key, id)
    enforce_cooldown("download", user.id, key, id, DOWNLOAD_COOLDOWN_SECONDS)
    return FileResponse(
        path=RESULTS_DIR / key / f"{artifact.value}.tflite",
        filename=f"{key}-{artifact.value}.tflite",
    )
