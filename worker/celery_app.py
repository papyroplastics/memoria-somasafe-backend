from celery import Celery

from common.celery_tasks import (
    CLEANUP_TASK,
    FED_AGG_TASK,
    FED_DISPATCH_TASK,
    HEAVY_QUEUE,
    LIGHT_QUEUE,
    QUANTIZE_TASK,
    SECURE_AGG_TASK,
    SECURE_SWEEP_TASK,
)
from common.config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_VISIBILITY_TIMEOUT_SECONDS,
    CLEANUP_INTERVAL_SECONDS,
    FED_AGG_INTERVAL_SECONDS,
    FED_LOCK_TTL_SECONDS,
    RESULT_TTL_SECONDS,
    SECURE_SWEEP_INTERVAL_SECONDS,
    WORKER_TASK_SOFT_TIME_LIMIT,
    WORKER_TASK_TIME_LIMIT,
)

if not (WORKER_TASK_SOFT_TIME_LIMIT < WORKER_TASK_TIME_LIMIT
        < FED_LOCK_TTL_SECONDS < CELERY_VISIBILITY_TIMEOUT_SECONDS):
    raise RuntimeError("expected WORKER_TASK_SOFT_TIME_LIMIT < WORKER_TASK_TIME_LIMIT < "
                       "FED_LOCK_TTL_SECONDS < CELERY_VISIBILITY_TIMEOUT_SECONDS")

app = Celery(
    "somasafe", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND,
    include=[
        "worker.tasks.quantize",
        "worker.tasks.aggregation",
        "worker.tasks.secure",
        "worker.tasks.cleanup",
    ],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    result_expires=RESULT_TTL_SECONDS,
    accept_content=["json"],
    task_default_queue=LIGHT_QUEUE,
    task_routes={
        QUANTIZE_TASK: {"queue": HEAVY_QUEUE},
        FED_AGG_TASK: {"queue": HEAVY_QUEUE},
        SECURE_AGG_TASK: {"queue": HEAVY_QUEUE},
        FED_DISPATCH_TASK: {"queue": LIGHT_QUEUE},
        SECURE_SWEEP_TASK: {"queue": LIGHT_QUEUE},
        CLEANUP_TASK: {"queue": LIGHT_QUEUE},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=WORKER_TASK_SOFT_TIME_LIMIT,
    task_time_limit=WORKER_TASK_TIME_LIMIT,
    broker_transport_options={"visibility_timeout": CELERY_VISIBILITY_TIMEOUT_SECONDS},
    beat_schedule={
        "cleanup-results": {
            "task": CLEANUP_TASK,
            "schedule": float(CLEANUP_INTERVAL_SECONDS),
        },
        "dispatch-aggregation": {
            "task": FED_DISPATCH_TASK,
            "schedule": float(FED_AGG_INTERVAL_SECONDS),
        },
        "secure-round-sweep": {
            "task": SECURE_SWEEP_TASK,
            "schedule": float(SECURE_SWEEP_INTERVAL_SECONDS),
        },
    },
)
