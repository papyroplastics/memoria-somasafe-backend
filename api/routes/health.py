from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlmodel import Session

from common.db import get_session
from common.redis import client as redis_client

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(session: Session = Depends(get_session)):
    try:
        session.execute(text("SELECT 1"))
        redis_client.ping()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ok"}
