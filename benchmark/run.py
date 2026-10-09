import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import redis
from sqlalchemy import func, select
from sqlmodel import Session

from benchmark import check, export, reset
from common.celery_tasks import HEAVY_QUEUE, LIGHT_QUEUE
from common.config import BROKER_URL, FED_AGG_INTERVAL_SECONDS, RESULTS_DIR, WORKER_METRICS_DIR
from common.db import (
    GlobalWeights,
    JobStatus,
    ModelVersion,
    QuantizationJob,
    SecurePartial,
    SecureSession,
    SecureSessionStatus,
    SubmissionType,
    User,
    engine,
)

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
HEAVY_MODEL = "mnist-mlp-heavy"
SMALL_WEIGHTS = 50_000
SPLITS = {
    "regular": lambda base, count: base != HEAVY_MODEL,
    "extreme": lambda base, count: base == HEAVY_MODEL,
    "small": lambda base, count: count < SMALL_WEIGHTS,
    "medium": lambda base, count: count >= SMALL_WEIGHTS and base != HEAVY_MODEL,
    "all": lambda base, count: True,
}
NATIVE = "native"
TOPOLOGIES = ("1x1", "2x2")
SECURE_SESSION_COLUMNS = ("id", "model_key", "status", "base_weights_id", "member_count",
                          "created_at", "sealed_at", "summing_at", "finished_at")
UNFINISHED_JOBS = (JobStatus.pending, JobStatus.running)
UNFINISHED_SESSIONS = (SecureSessionStatus.sealed, SecureSessionStatus.summing)
SETTLE_POLL_SECONDS = 5


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def parse_stages(spec: str) -> list[tuple[float, int]]:
    try:
        stages = [(float(seconds), int(active)) for seconds, active in
                  (stage.split(":") for stage in spec.split(","))]
    except ValueError:
        raise SystemExit(f"invalid --stages '{spec}', expected <seconds>:<active users>,...")
    if not stages or any(seconds <= 0 or active < 0 for seconds, active in stages):
        raise SystemExit(f"invalid --stages '{spec}'")
    return stages


def latest_versions() -> dict[str, tuple[SubmissionType, int]]:
    with Session(engine) as session:
        rows = session.execute(
            select(ModelVersion.model_key, ModelVersion.submission_type, ModelVersion.weight_count)
            .distinct(ModelVersion.model_key)
            .order_by(ModelVersion.model_key, ModelVersion.version.desc())).all()  # type: ignore[attr-defined]
    return {key: (kind, count) for key, kind, count in rows}


def select_models(split: str, submission: str, explicit: str) -> dict[str, int]:
    latest = latest_versions()
    if explicit:
        keys = [key for key in explicit.split(",") if key]
        missing = [key for key in keys if key not in latest]
        if missing:
            raise SystemExit(f"models not seeded: {', '.join(missing)}")
        return {key: latest[key][1] for key in keys}
    models = {}
    for key, (kind, count) in latest.items():
        base = key.removesuffix(f"-{kind.value}")
        if not SPLITS[split](base, count):
            continue
        if (key == base) if submission == NATIVE else (kind.value == submission):
            models[key] = count
    if not models:
        raise SystemExit(f"no seeded model matches split '{split}' and submission type '{submission}'")
    return models


def check_users(users: int, stride: int) -> None:
    highest = f"test_{stride * math.ceil(users / stride)}"
    with Session(engine) as session:
        if session.execute(select(User.id).where(User.username == highest)).first() is None:
            raise SystemExit(f"user {highest} is missing, seed more with "
                             f"`uv run -m scripts.system.seed_db --test-users N`")


def idle(broker: redis.Redis) -> bool:
    queued = sum(broker.llen(queue) for queue in (LIGHT_QUEUE, HEAVY_QUEUE)) + broker.hlen("unacked")
    if queued:
        return False
    with Session(engine) as session:
        jobs = session.execute(select(func.count()).select_from(QuantizationJob)
                               .where(QuantizationJob.status.in_(UNFINISHED_JOBS))).scalar_one()  # type: ignore[attr-defined]
        sessions = session.execute(select(func.count()).select_from(SecureSession)
                                   .where(SecureSession.status.in_(UNFINISHED_SESSIONS))).scalar_one()  # type: ignore[attr-defined]
        partials = session.execute(
            select(func.count()).select_from(SecurePartial)
            .join(SecureSession, SecureSession.id == SecurePartial.session_id)  # type: ignore[arg-type]
            .where(~select(GlobalWeights.id)
                   .where(GlobalWeights.parent_weights_id == SecureSession.base_weights_id,
                          GlobalWeights.valid == True)
                   .exists())).scalar_one()
    return jobs == 0 and sessions == 0 and partials == 0


def settle(load_end: float, interval: float, timeout: float) -> bool:
    broker = redis.from_url(BROKER_URL)
    while time.time() < load_end + timeout:
        if time.time() >= load_end + interval and idle(broker):
            return True
        time.sleep(SETTLE_POLL_SECONDS)
    return False


def save_secure_sessions(path: Path) -> int:
    columns = [getattr(SecureSession, column) for column in SECURE_SESSION_COLUMNS]
    with Session(engine) as session:
        rows = session.execute(select(*columns).order_by(SecureSession.id)).all()  # type: ignore[arg-type]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SECURE_SESSION_COLUMNS)
        for row in rows:
            writer.writerow([value.isoformat() if isinstance(value, datetime)
                             else getattr(value, "value", value) for value in row])
    return len(rows)


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the stack, drive one Locust run, collect and check it")
    parser.add_argument("topology", choices=TOPOLOGIES, help="topology the stack was brought up with")
    parser.add_argument("--run-id", help="defaults to <timestamp>-<topology>-<split>-<submission>")
    parser.add_argument("--split", choices=SPLITS, default="regular")
    parser.add_argument("--models", default="", help="comma-separated model keys, overrides --split/--submission")
    parser.add_argument("--submission", choices=[NATIVE, *(kind.value for kind in SubmissionType)],
                        default=NATIVE, help="submission type every model is exercised with")
    parser.add_argument("--stages", default="120:5,60:10,60:20,60:30,60:5",
                        help="<seconds>:<active users> stages; every user logs in during the first one")
    parser.add_argument("--spawn-rate", type=float, default=2.0, help="users (logins) per second")
    parser.add_argument("--processes", type=int, default=2, help="Locust worker processes")
    parser.add_argument("--session-failure", type=float, default=0.1,
                        help="probability of a secure session failing on the seal timeout")
    parser.add_argument("--settle-timeout", type=float, help="seconds, defaults to three aggregation intervals")
    parser.add_argument("--host", default=os.environ.get("BENCH_API_URL", "http://localhost:8000"))
    parser.add_argument("--prometheus", default=os.environ.get("BENCH_PROMETHEUS_URL", "http://localhost:9090"))
    parser.add_argument("--metrics-dir", default=WORKER_METRICS_DIR,
                        help="directory holding the worker metrics, empty to skip copying it")
    args = parser.parse_args()

    topology = args.topology
    stages = parse_stages(args.stages)
    users = max(active for _, active in stages)
    login_seconds = users / args.spawn_rate
    if stages[0][0] < login_seconds:
        raise SystemExit(f"the first stage lasts {stages[0][0]:g}s but {users} logins at "
                         f"{args.spawn_rate:g}/s take {login_seconds:g}s")
    models = select_models(args.split, args.submission, args.models)
    check_users(users, args.processes)
    interval = float(FED_AGG_INTERVAL_SECONDS)

    label = "custom" if args.models else args.split
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{topology}-{label}-{args.submission}"
    run_dir = RESULTS_DIR / "benchmark" / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists")
    run_dir.mkdir(parents=True)
    print(f"run {run_id}: {topology}, models " + ", ".join(f"{key} ({count})" for key, count in models.items()))

    reset.main()

    schedule_start = time.time()
    boundaries = [schedule_start]
    for seconds, _ in stages:
        boundaries.append(boundaries[-1] + seconds)
    manifest = {
        "run_id": run_id,
        "topology": topology,
        "split": label,
        "submission_type": args.submission,
        "models": models,
        "load": {"users": users, "spawn_rate": args.spawn_rate, "stages": args.stages,
                 "processes": args.processes, "interval": interval,
                 "session_failure": args.session_failure, "host": args.host},
        "stages": [{"start": iso(begin), "end": iso(end), "active": active}
                   for (_, active), begin, end in zip(stages, boundaries, boundaries[1:])],
        "start": iso(schedule_start),
        "load_end": None,
        "end": None,
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    locust = subprocess.run([
        sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless", "--only-summary",
        "--host", args.host, "--users", str(users), "--spawn-rate", str(args.spawn_rate),
        "--run-time", f"{math.ceil(boundaries[-1] - schedule_start)}s", "--processes", str(args.processes),
        "--run-dir", str(run_dir), "--models", ",".join(models), "--interval", str(interval),
        "--user-stride", str(args.processes), "--session-failure", str(args.session_failure),
        "--stages", args.stages, "--schedule-start", str(schedule_start),
    ])
    load_end = time.time()
    manifest["load_end"] = iso(load_end)
    manifest["locust_exit_code"] = locust.returncode
    write_manifest(manifest_path, manifest)

    print("waiting for the queues and secure sessions to settle")
    manifest["settled"] = settle(load_end, interval, args.settle_timeout or 3 * interval)
    manifest["end"] = iso(time.time())
    write_manifest(manifest_path, manifest)
    if not manifest["settled"]:
        print("the stack did not settle before the timeout")

    print(f"saved {save_secure_sessions(run_dir / 'secure_sessions.csv')} secure session(s)")
    if args.metrics_dir:
        shutil.copytree(args.metrics_dir, run_dir / "worker_metrics", dirs_exist_ok=True)
    export.dump(run_dir, args.prometheus)
    check.run(run_dir, args.prometheus)
    print(f"run artifacts in {run_dir}, plot with `uv run -m benchmark.export {run_id}`")


if __name__ == "__main__":
    main()
