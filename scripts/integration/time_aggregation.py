"""Time one federated aggregation round end to end (report Sec. 5.5, tab:costo-borde).

Measures the wall time from enqueuing the ``federated_aggregation`` task to its result
landing, so the figure is the round itself and not the interpreter start-up."""

import sys
import time

from common.celery_tasks import FED_AGG_TASK
from worker.celery_app import app

from scripts.common.api import wait_for_aggregation

key = sys.argv[1] if len(sys.argv) > 1 else "feature-ae"

start = time.perf_counter()
summary = wait_for_aggregation(app.send_task(FED_AGG_TASK, args=[key]), key)
elapsed = time.perf_counter() - start

print(f"{key}: {summary}")
print(f"AGG_WALL_SECONDS={elapsed:.3f}")
