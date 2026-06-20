from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.db import init_db
from .auth import router as auth_router
from .model import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(auth_router)
app.include_router(router)
