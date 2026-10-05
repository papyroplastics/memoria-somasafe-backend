import argparse
import csv
import json
import math
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import select
from sqlmodel import Session

from benchmark.scripts import reset
from common.config import RESULTS_DIR
from common.db import ModelVersion, SecureRound, SubmissionType, User, engine

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
HEAVY_MODEL = "mnist-mlp-heavy"
SPLITS = {"regular": lambda base: base != HEAVY_MODEL, "extreme": lambda base: base == HEAVY_MODEL}
NATIVE = "native"
TOPOLOGIES = ("1x1", "2x2")
SECURE_ROUND_COLUMNS = ("id", "model_key", "status", "member_count", "created_at",
                        "sealed_at", "aggregating_at", "finished_at")
PRIVATE_CONFIG = ("POSTGRES_", "SEED_", "REDIS_HOST", "BROKER_HOST")


def now() -> str:
    return datetime.now(UTC).isoformat()


def select_models(split: str, submission: str) -> list[str]:
    with Session(engine) as session:
        latest = session.execute(
            select(ModelVersion.model_key, ModelVersion.submission_type)
            .distinct(ModelVersion.model_key)
            .order_by(ModelVersion.model_key, ModelVersion.version.desc())).all()  # type: ignore[attr-defined]
    keys = []
    for key, kind in latest:
        base = key.removesuffix(f"-{kind.value}")
        if not SPLITS[split](base):
            continue
        if (key == base) if submission == NATIVE else (kind.value == submission):
            keys.append(key)
    if not keys:
        raise SystemExit(f"no seeded model matches split '{split}' and submission type '{submission}'")
    return keys


def check_users(users: int, stride: int) -> None:
    highest = f"test_{stride * math.ceil(users / stride)}"
    with Session(engine) as session:
        if session.execute(select(User.id).where(User.username == highest)).first() is None:
            raise SystemExit(f"user {highest} is missing, seed more with "
                             f"`uv run -m scripts.system.seed_db --test-users N`")


def config_overrides(env_file: Path) -> dict[str, str | None]:
    return {key: value for key, value in dotenv_values(env_file).items()
            if not key.startswith(PRIVATE_CONFIG)}


def save_secure_rounds(path: Path) -> int:
    columns = [getattr(SecureRound, column) for column in SECURE_ROUND_COLUMNS]
    with Session(engine) as session:
        rows = session.execute(select(*columns).order_by(SecureRound.id)).all()  # type: ignore[arg-type]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SECURE_ROUND_COLUMNS)
        for row in rows:
            writer.writerow([value.isoformat() if isinstance(value, datetime)
                             else getattr(value, "value", value) for value in row])
    return len(rows)


def copy_worker_metrics(volume: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    export = subprocess.Popen(["podman", "volume", "export", volume], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=export.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    if export.wait() != 0:
        raise SystemExit(f"could not export volume {volume}")


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the stack, drive one Locust run and collect its artifacts")
    parser.add_argument("topology", choices=TOPOLOGIES, help="topology the stack was brought up with")
    parser.add_argument("--run-id", help="defaults to <timestamp>-<topology>-<split>-<submission>")
    parser.add_argument("--split", choices=SPLITS, default="regular")
    parser.add_argument("--submission", choices=[NATIVE, *(kind.value for kind in SubmissionType)],
                        default=NATIVE, help="submission type every model is exercised with")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--spawn-rate", type=float, default=2.0, help="users (logins) per second")
    parser.add_argument("--duration", type=int, default=180, help="seconds")
    parser.add_argument("--processes", type=int, default=2, help="Locust worker processes")
    parser.add_argument("--round-failure", type=float, default=0.1,
                        help="probability of a secure round failing on the seal timeout")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--env-file", type=Path, default=Path("benchmark/prod.env"))
    parser.add_argument("--metrics-volume", default="backend_worker_metrics",
                        help="podman volume holding the worker metrics, empty to skip copying it")
    args = parser.parse_args()

    topology = args.topology
    models = select_models(args.split, args.submission)
    check_users(args.users, args.processes)
    config = config_overrides(args.env_file)
    interval = float(config.get("FED_AGG_INTERVAL_SECONDS") or 86400)

    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{topology}-{args.split}-{args.submission}"
    run_dir = RESULTS_DIR / "benchmark" / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists")
    run_dir.mkdir(parents=True)
    print(f"run {run_id}: {topology}, models {', '.join(models)}")

    reset.main()

    manifest = {
        "run_id": run_id,
        "topology": topology,
        "split": args.split,
        "submission_type": args.submission,
        "models": models,
        "load": {"users": args.users, "spawn_rate": args.spawn_rate, "duration": args.duration,
                 "processes": args.processes, "interval": interval,
                 "round_failure": args.round_failure, "host": args.host},
        "config": config,
        "start": now(),
        "end": None,
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    locust = subprocess.run([
        sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless", "--only-summary",
        "--host", args.host, "--users", str(args.users), "--spawn-rate", str(args.spawn_rate),
        "--run-time", f"{args.duration}s", "--processes", str(args.processes),
        "--run-dir", str(run_dir), "--models", ",".join(models), "--interval", str(interval),
        "--user-stride", str(args.processes), "--round-failure", str(args.round_failure),
    ])
    manifest["end"] = now()
    manifest["locust_exit_code"] = locust.returncode
    write_manifest(manifest_path, manifest)

    print(f"saved {save_secure_rounds(run_dir / 'secure_rounds.csv')} secure round(s)")
    if args.metrics_volume:
        copy_worker_metrics(args.metrics_volume, run_dir / "worker_metrics")
    print(f"run artifacts in {run_dir}, export with `uv run -m benchmark.scripts.export {run_id}`")


if __name__ == "__main__":
    main()
