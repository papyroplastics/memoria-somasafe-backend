import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlmodel import Session

from worker.celery_app import app as celery_app

from common.config import RESULTS_DIR
from common.db import JobStatus, QuantizationJob, WeightSubmission, get_session, utcnow
from common.models import model_list, models

router = APIRouter(prefix="/model")

model_file = "trainable.tflite"
QUANTIZE_TASK = "worker.tasks.quantize_submission"


class ModelWeights(BaseModel):
    parameters: list[float]


def require_model(key: str, id: int) -> dict:
    meta = models.get(key)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found")
    if id != meta["model_id"]:
        raise HTTPException(status_code=404, detail=f"Model '{key}' version {id} not found")
    return meta


@router.get("/list")
async def list_models():
    return model_list


@router.get("/trainable/{key}/{id}", response_class=FileResponse)
async def get_model(key: str, id: int):
    require_model(key, id)
    return FileResponse(path=RESULTS_DIR / key / model_file, filename=f"{key}.tflite")


@router.post("/quantize/{key}/{id}", status_code=202)
def quantize_model(key: str, id: int, weights: ModelWeights,
                   session: Session = Depends(get_session)):
    require_model(key, id)

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
def quantize_result(job_id: uuid.UUID, session: Session = Depends(get_session)):
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
