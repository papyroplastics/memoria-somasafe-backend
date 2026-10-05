import base64
import csv
import logging
import os
import random
import time
from array import array
from functools import cache
from pathlib import Path
from urllib.parse import urlencode

from locust import FastHttpUser, constant_pacing, events, task
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
    group.add_argument("--round-failure", type=float, default=0.1,
                       help="probability of a secure round failing, its lowest user id never submits")


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
def dense_body(weight_count: int) -> bytes:
    return (array("f", [DELTA]) * weight_count).tobytes()


@cache
def masked_body(weight_count: int) -> bytes:
    return (array("I", [1]) * weight_count).tobytes()


class Client(FastHttpUser):
    spawned = 0

    def wait_time(self) -> float:
        return constant_pacing(self.options.interval)(self)

    def on_start(self) -> None:
        self.options = self.environment.parsed_options
        index = Client.spawned
        Client.spawned += 1
        number = 1 + max(self.environment.runner.worker_index, 0) + index * self.options.user_stride
        self.username = f"test_{number}"
        self.password = self.username if number <= TEST_SUBJECTS else SHARED_PASSWORD
        self.headers: dict[str, str] | None = None
        self.last_start: float | None = None
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
        lag = 0.0 if self.last_start is None else max(0.0, started - self.last_start - self.options.interval)
        self.last_start = started
        events.request.fire(request_type="LAG", name=SCHEDULE_LAG, response_time=lag * 1000,
                            response_length=0, response=None, context={}, exception=None,
                            start_time=started, url="")
        if self.headers is None and not self.login():
            return

        jobs, rounds = [], []
        for model in self.models():
            key, kind = model["key"], model["submission_type"]
            weights_id = self.download(key)
            if self.headers is None:
                return
            if kind == "secure":
                rounds.extend(self.join(key))
            elif weights_id is not None:
                jobs.extend(self.submit(key, kind, weights_id, model["weight_count"]))
        self.settle(jobs, rounds, started + self.options.interval * POLL_WINDOW)

    def download(self, key: str) -> int | None:
        resp = self.call("GET", f"/model/weights/{key}", "weights")
        weights_id = resp.headers.get("X-Weights-ID") if resp.status_code == 200 else None
        for artifact in ARTIFACTS:
            self.call("GET", f"/model/download/{artifact}/{key}", "download/{artifact}")
        return int(weights_id) if weights_id is not None else None

    def submit(self, key: str, kind: str, weights_id: int, weight_count: int) -> list[str]:
        resp = self.call("POST", f"/model/submit/{kind}/{key}/{weights_id}", f"submit/{kind}",
                         data=dense_body(weight_count), headers=OCTET)
        return [resp.json()["job_id"]] if kind == "quantize" and resp.status_code == 202 else []

    def join(self, key: str) -> list[tuple[str, int, int]]:
        ka_public_key = base64.b64encode(b"\x04" + os.urandom(64)).decode()
        resp = self.call("POST", f"/model/secure/join/{key}", "secure/join",
                         json={"ka_public_key": ka_public_key})
        if resp.status_code != 202:
            return []
        joined = resp.json()
        return [(key, joined["round_id"], joined["user_id"])]

    def settle(self, jobs: list[str], rounds: list[tuple[str, int, int]], deadline: float) -> None:
        while (jobs or rounds) and self.headers is not None:
            jobs = [job_id for job_id in jobs if self.quantizing(job_id)]
            rounds = [joined for joined in rounds if self.round_open(*joined)]
            if not (jobs or rounds) or time.time() + POLL_SECONDS > deadline:
                return
            time.sleep(POLL_SECONDS)

    def quantizing(self, job_id: str) -> bool:
        return self.call("GET", f"/model/quantize/result/{job_id}", "quantize/result").status_code == 202

    def round_open(self, key: str, round_id: int, user_id: int) -> bool:
        with self.client.get(f"/model/secure/round/{round_id}", name="secure/round", headers=self.headers,
                             catch_response=True) as resp:
            if resp.status_code == 409 and resp.json().get("detail", "").startswith("Round is open"):
                resp.success()
                return True
            if resp.status_code == 401:
                self.headers = None
            if resp.status_code != 200:
                return False
            descriptor = resp.json()
        if not self.drops(key, round_id, user_id, descriptor["roster"]):
            self.call("POST", f"/model/secure/submit/{round_id}", "secure/submit",
                      data=masked_body(descriptor["weight_count"]), headers=OCTET)
        return False

    def drops(self, key: str, round_id: int, user_id: int, roster: list[dict]) -> bool:
        if user_id != min(member["user_id"] for member in roster):
            return False
        return random.Random(f"{key}:{round_id}").random() < self.options.round_failure
