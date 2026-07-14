from celery import Celery

from common.celery_tasks import CLEANUP_TASK, FED_AGG_TASK
from common.config import (
    REDIS_URL,
    CLEANUP_INTERVAL_SECONDS,
    FED_AGG_INTERVAL_SECONDS,
    RESULT_TTL_SECONDS,
)

app = Celery("somasafe", broker=REDIS_URL, backend=REDIS_URL)

# Job state lives in PostgreSQL (see worker.tasks). The Redis result backend exists
# only so callers can await a task and read its return value (e.g. the aggregation
# summary) instead of polling the DB; results expire quickly to avoid buildup.
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    result_expires=RESULT_TTL_SECONDS,
    accept_content=["json"],
    beat_schedule={
        "cleanup-results": {
            "task": CLEANUP_TASK,
            "schedule": float(CLEANUP_INTERVAL_SECONDS),
        },
        "federated-aggregation": {
            "task": FED_AGG_TASK,
            "schedule": float(FED_AGG_INTERVAL_SECONDS),
        },
    },
)

app.autodiscover_tasks(["worker"])
