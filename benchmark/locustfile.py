import base64
import csv
import logging
import math
import os
import random
import time
from array import array
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from urllib.parse import urlencode

from locust import FastHttpUser, events, task
from locust.runners import MasterRunner

SCHEDULE_LAG = "schedule_lag"
LOG_HEADER = ("timestamp", "name", "status", "latency_ms", "response_bytes")
TEST_SUBJECTS = 15
SHARED_PASSWORD = "test"
DELTA = 1e-4
POLL_SECONDS = 2.0
POLL_WINDOW = 0.8
ARTIFACTS = ("trainable", "quantized")
FORM = {"Content-Type": "application/x-www-form-urlencoded"}
OCTET = {"Content-Type": "application/octet-stream"}
PARK_SECONDS = 1.0
GOLDEN = 0.6180339887
OPEN = "Session is open"

log = logging.getLogger(__name__)
served: list[dict] | None = None


@events.init_command_line_parser.add_listener
def add_arguments(parser) -> None:
    group = parser.add_argument_group("benchmark")
    group.add_argument("--run-dir", default="results/benchmark/adhoc",
                       help="directory the requests_<n>.csv logs are written to")
    group.add_argument("--models", default="",
                       help="comma-separated model keys to exercise, every served model when empty")
    group.add_argument("--interval", type=float, default=60.0,
                       help="aggregation interval in seconds, iterations are paced at it")
    group.add_argument("--user-stride", type=int, default=1,
                       help="Locust worker processes across all hosts, so test_N users never collide")
    group.add_argument("--session-failure", type=float, default=0.1,
                       help="probability of a secure session failing, its lowest user id never submits")
    group.add_argument("--stages", default="",
                       help="comma-separated <seconds>:<active users> stages, every user active when empty")
    group.add_argument("--schedule-start", type=float, default=0.0,
                       help="unix time the stages count from, shared by every Locust process")


class RequestLog:
    file = None
    writer = None

    @classmethod
    def row(cls, environment, values: tuple) -> None:
        if cls.writer is None:
            if isinstance(environment.runner, MasterRunner):
                return
            run_dir = Path(environment.parsed_options.run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            index = max(environment.runner.worker_index, 0)
            cls.file = (run_dir / f"requests_{index}.csv").open("w", newline="")
            cls.writer = csv.writer(cls.file)
            cls.writer.writerow(LOG_HEADER)
        cls.writer.writerow(values)

    @classmethod
    def flush(cls, **_) -> None:
        if cls.file is not None:
            cls.file.flush()

    @classmethod
    def close(cls, **_) -> None:
        if cls.file is not None:
            cls.file.close()
            cls.file = cls.writer = None


environment_ref = None


@events.init.add_listener
def on_init(environment, **_) -> None:
    global environment_ref
    environment_ref = environment


@events.request.add_listener
def on_request(name, response_time, response_length, response=None, start_time=None, **_) -> None:
    status = getattr(response, "status_code", None) or 0
    RequestLog.row(environment_ref, (f"{start_time:.3f}", name, status,
                                     round(response_time, 2), response_length or 0))


events.test_stop.add_listener(RequestLog.flush)
events.quitting.add_listener(RequestLog.close)


@cache
def stage_ends(spec: str) -> list[tuple[float, int]]:
    ends, elapsed = [], 0.0
    for stage in filter(None, spec.split(",")):
        seconds, active = stage.split(":")
        elapsed += float(seconds)
        ends.append((elapsed, int(active)))
    return ends


def active_users(options, now: float) -> float:
    if not options.stages:
        return math.inf
    elapsed = now - options.schedule_start
    if elapsed < 0:
        return 0
    for end, active in stage_ends(options.stages):
        if elapsed < end:
            return active
    return 0


@cache
def dense_body(weight_count: int) -> bytes:
    return (array("f", [DELTA]) * weight_count).tobytes()


@cache
def masked_body(weight_count: int) -> bytes:
    return (array("I", [1]) * weight_count).tobytes()


@dataclass(frozen=True)
class Seat:
    model_key: str
    weights_id: int
    session_id: int
    user_id: int
    submitted: bool = False


class Client(FastHttpUser):
    spawned = 0

    def wait_time(self) -> float:
        if self.next_start is None:
            return PARK_SECONDS
        return max(0.0, self.next_start - time.time())

    def on_start(self) -> None:
        self.options = self.environment.parsed_options
        index = Client.spawned
        Client.spawned += 1
        self.number = 1 + max(self.environment.runner.worker_index, 0) + index * self.options.user_stride
        self.username = f"test_{self.number}"
        self.password = self.username if self.number <= TEST_SUBJECTS else SHARED_PASSWORD
        self.offset = (self.number * GOLDEN) % 1 * self.options.interval
        self.headers: dict[str, str] | None = None
        self.next_start: float | None = None
        self.weights: dict[str, int] = {}
        self.seats: dict[str, Seat] = {}
        self.login()

    def login(self) -> bool:
        body = urlencode({"username": self.username, "password": self.password})
        with self.client.post("/auth/token", data=body, headers=FORM, name="auth/token",
                              catch_response=True) as resp:
            if resp.status_code == 200:
                self.headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
        return self.headers is not None

    def call(self, method: str, path: str, name: str, **kwargs):
        headers = (self.headers or {}) | kwargs.pop("headers", {})
        resp = self.client.request(method, path, name=name, headers=headers, **kwargs)
        if resp.status_code == 401:
            self.headers = None
        return resp

    def models(self) -> list[dict]:
        global served
        if served is None:
            resp = self.call("GET", "/model/list", "model/list")
            if resp.status_code != 200:
                return []
            wanted = [key for key in self.options.models.split(",") if key]
            listed = {model["key"]: model for model in resp.json()}
            missing = [key for key in wanted if key not in listed]
            if missing:
                log.warning("models not served: %s", ", ".join(missing))
            served = [listed[key] for key in wanted if key in listed] if wanted else list(listed.values())
        return served

    @task
    def iteration(self) -> None:
        started = time.time()
        if self.number > active_users(self.options, started):
            self.next_start = None
            return
        if self.next_start is None:
            self.next_start = started + self.offset
            return
        if started < self.next_start:
            return
        lag = started - self.next_start
        self.next_start = started + self.options.interval
        events.request.fire(request_type="LAG", name=SCHEDULE_LAG, response_time=lag * 1000,
                            response_length=0, response=None, context={}, exception=None,
                            start_time=started, url="")
        if self.headers is None and not self.login():
            return

        jobs, sessions = [], []
        for model in self.models():
            key, kind = model["key"], model["submission_type"]
            weights_id = self.download(key)
            if self.headers is None:
                return
            if kind == "secure":
                held = self.seats.pop(key, None)
                sessions.extend([held] if held else self.join(key, self.weights.get(key)))
            elif weights_id is not None:
                jobs.extend(self.submit(key, kind, weights_id, model["weight_count"]))
        self.settle(jobs, sessions, started + self.options.interval * POLL_WINDOW)

    def download(self, key: str) -> int | None:
        resp = self.call("GET", f"/model/weights/{key}", "weights")
        weights_id = int(resp.headers["X-Weights-ID"]) if resp.status_code == 200 else None
        for artifact in ARTIFACTS:
            self.call("GET", f"/model/download/{artifact}/{key}", "download/{artifact}")
        if weights_id is not None:
            self.weights[key] = weights_id
        return weights_id

    def submit(self, key: str, kind: str, weights_id: int, weight_count: int) -> list[str]:
        resp = self.call("POST", f"/model/submit/{kind}/{key}/{weights_id}", f"submit/{kind}",
                         data=dense_body(weight_count), headers=OCTET)
        return [resp.json()["job_id"]] if kind == "quantize" and resp.status_code == 202 else []

    def join(self, key: str, weights_id: int | None) -> list[Seat]:
        if weights_id is None:
            return []
        ka_public_key = base64.b64encode(b"\x04" + os.urandom(64)).decode()
        resp = self.call("POST", f"/model/secure/join/{key}/{weights_id}", "secure/join",
                         json={"ka_public_key": ka_public_key})
        if resp.status_code != 202:
            return []
        joined = resp.json()
        return [Seat(key, weights_id, joined["session_id"], joined["user_id"])]

    def settle(self, jobs: list[str], sessions: list[Seat], deadline: float) -> None:
        while (jobs or sessions) and self.headers is not None:
            jobs = [job_id for job_id in jobs if self.quantizing(job_id)]
            sessions = [after for seat in sessions for after in self.session_step(seat)]
            if not (jobs or sessions) or time.time() + POLL_SECONDS > deadline:
                break
            time.sleep(POLL_SECONDS)
        self.seats.update((seat.model_key, seat) for seat in sessions)

    def quantizing(self, job_id: str) -> bool:
        return self.call("GET", f"/model/quantize/result/{job_id}", "quantize/result").status_code == 202

    def session_step(self, seat: Seat) -> list[Seat]:
        with self.client.get(f"/model/secure/session/{seat.session_id}", name="secure/session",
                             headers=self.headers, catch_response=True) as resp:
            if resp.status_code in (404, 409):
                resp.success()
            if resp.status_code == 401:
                self.headers = None
            if resp.status_code == 404:
                return self.join(seat.model_key, self.weights.get(seat.model_key))
            if resp.status_code == 409 and resp.json().get("detail", "").startswith(OPEN):
                return [seat]
            if resp.status_code != 200:
                return []
            descriptor = resp.json()
        if seat.submitted:
            return [seat]
        if self.drops(seat, descriptor["roster"]):
            return []
        resp = self.call("POST", f"/model/secure/submit/{seat.session_id}", "secure/submit",
                         data=masked_body(descriptor["weight_count"]), headers=OCTET)
        return [replace(seat, submitted=True)] if resp.status_code == 202 else []

    def drops(self, seat: Seat, roster: list[dict]) -> bool:
        if seat.user_id != min(member["user_id"] for member in roster):
            return False
        return random.Random(f"{seat.model_key}:{seat.session_id}").random() < self.options.session_failure
