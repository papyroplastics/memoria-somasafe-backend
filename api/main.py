from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.db import init_db
from .routes.auth import router as auth_router
from .routes.device import router as device_router
from .routes.model import router
from .routes import secure as secure
from .routes.ota import router as ota_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(auth_router)
app.include_router(device_router)
app.include_router(router)
app.include_router(ota_router)
