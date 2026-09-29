import csv
import logging
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from common.config import WORKER_METRICS_DIR

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

log = logging.getLogger(__name__)


class Timer[S: StrEnum]:
    def __init__(self, *stages: S):
        self._names = tuple(stage.value for stage in stages)
        if len(set(self._names)) != len(self._names):
            raise ValueError(f"duplicate stage names in {self._names}")
        self._seconds: dict[str, float] = {}
        self._start = time.perf_counter()
        self.started_at = datetime.now(UTC)

    @contextmanager
    def __call__(self, stage: S) -> Iterator[None]:
        if stage.value not in self._names:
            raise ValueError(f"stage '{stage.value}' was not declared")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._seconds[stage.value] = self._seconds.get(stage.value, 0.0) + elapsed

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def seconds(self) -> dict[str, float | None]:
        return {name: round(self._seconds[name], 6) if name in self._seconds else None
                for name in self._names}


@cache
def _directory(pid: int) -> Path | None:
    if WORKER_METRICS_DIR is None:
        return None
    path = Path(WORKER_METRICS_DIR) / f"{socket.gethostname()}-{pid}"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("worker metrics disabled: %s", exc)
        return None
    return path


def write(task: str, record: "DataclassInstance", timer: Timer) -> dict[str, object]:
    row = {"started_at": timer.started_at.isoformat(),
           "total_seconds": round(timer.elapsed(), 6),
           **asdict(record), **timer.seconds()}
    directory = _directory(os.getpid())
    if directory is not None:
        try:
            with (directory / f"{task}.csv").open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            log.warning("could not write %s metrics: %s", task, exc)
    return row
