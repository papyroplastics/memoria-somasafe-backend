import logging
import time
from contextlib import contextmanager

log = logging.getLogger("worker.phases")


class Phases:
    def __init__(self, **labels):
        self.labels = " ".join(f"{key}={value}" for key, value in labels.items())
        self.timings: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[name] = self.timings.get(name, 0.0) + elapsed
            log.info("phase=%s seconds=%.4f %s", name, elapsed, self.labels)
