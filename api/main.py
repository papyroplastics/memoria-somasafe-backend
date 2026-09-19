from fastapi import FastAPI

from .routes.auth import router as auth_router
from .routes.device import router as device_router
from .routes.model import router
from .routes import secure as secure
from .routes.ota import router as ota_router

app = FastAPI()
app.include_router(auth_router)
app.include_router(device_router)
app.include_router(router)
app.include_router(ota_router)
