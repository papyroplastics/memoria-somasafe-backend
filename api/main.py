from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator, metrics

from .routes.auth import router as auth_router
from .routes.device import router as device_router
from .routes.health import router as health_router
from .routes.model import router
from .routes import secure as secure
from .routes.ota import router as ota_router

app = FastAPI()
app.include_router(auth_router)
app.include_router(device_router)
app.include_router(health_router)
app.include_router(router)
app.include_router(ota_router)

# should_group_status_codes=False: 429 is a first-class result, not noise to
# fold into a "4xx" bucket.
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_instrument_requests_inprogress=True,
    inprogress_name="http_requests_in_progress",
    excluded_handlers=["/healthz", "/readyz", "/metrics"],
)
instrumentator.add(
    metrics.latency(),
    metrics.requests(),
    metrics.request_size(),
    metrics.response_size(),
)
instrumentator.instrument(app).expose(app)
