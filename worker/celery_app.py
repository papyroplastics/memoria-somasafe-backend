from celery import Celery

from common.config import BROKER_URL, CLEANUP_INTERVAL_SECONDS

app = Celery("somasafe", broker=BROKER_URL)

# Job state lives in PostgreSQL (see worker.tasks), so no Celery result backend.
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    beat_schedule={
        "cleanup-results": {
            "task": "worker.tasks.cleanup_results",
            "schedule": float(CLEANUP_INTERVAL_SECONDS),
        },
    },
)

app.autodiscover_tasks(["worker"])
