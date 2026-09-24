"""Time one federated aggregation round end to end (report Sec. 5.5, tab:costo-borde). """

import sys
import time

from worker.celery_app import app

from scripts.common.api import wait_for_aggregation

key = sys.argv[1] if len(sys.argv) > 1 else "feature-ae"

start = time.perf_counter()
result = wait_for_aggregation(app, key)
elapsed = time.perf_counter() - start

print(f"{key}: {result['outcome']} ({result['detail']})")
for phase, seconds in result["timings"].items():
    print(f"  {phase}: {seconds:.3f}s")
print(f"AGG_WALL_SECONDS={elapsed:.3f}")
