import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results")) / "benchmark"
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
SUBMISSION_TYPES = ("raw", "quantize", "secure")
TOPOLOGIES = ("1x1", "2x2")
TEST_SUBJECTS = 15
SHARED_PASSWORD = "test"


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


def login(host: str, number: int) -> str | None:
    username = f"test_{number}"
    password = username if number <= TEST_SUBJECTS else SHARED_PASSWORD
    body = urllib.parse.urlencode({"username": username, "password": password}).encode()
    try:
        with urllib.request.urlopen(f"{host}/auth/token", data=body, timeout=30) as resp:
            return json.load(resp)["access_token"]
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return None
        raise


def served_models(host: str) -> dict[str, tuple[str, int]]:
    token = login(host, 1)
    if token is None:
        raise SystemExit("cannot log in as test_1, seed the test users first")
    request = urllib.request.Request(f"{host}/model/list", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=30) as resp:
        return {model["key"]: (model["submission_type"], model["weight_count"]) for model in json.load(resp)}


def select_models(host: str, split: str, submission: str, explicit: str) -> dict[str, int]:
    served = served_models(host)
    if explicit:
        keys = [key for key in explicit.split(",") if key]
        missing = [key for key in keys if key not in served]
        if missing:
            raise SystemExit(f"models not served: {', '.join(missing)}")
        return {key: served[key][1] for key in keys}
    models = {}
    for key, (kind, count) in served.items():
        base = key.removesuffix(f"-{kind}")
        if not SPLITS[split](base, count):
            continue
        if (key == base) if submission == NATIVE else (kind == submission):
            models[key] = count
    if not models:
        raise SystemExit(f"no served model matches split '{split}' and submission type '{submission}'")
    return models


def check_users(host: str, users: int, stride: int) -> None:
    highest = stride * math.ceil(users / stride)
    if login(host, highest) is None:
        raise SystemExit(f"user test_{highest} is missing, seed more with `--test-users N`")


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive one staged Locust run and write its manifest")
    parser.add_argument("topology", choices=TOPOLOGIES, help="topology the stack was brought up with")
    parser.add_argument("--run-id", help="defaults to <timestamp>-<topology>-<split>-<submission>")
    parser.add_argument("--split", choices=SPLITS, default="regular")
    parser.add_argument("--models", default="", help="comma-separated model keys, overrides --split/--submission")
    parser.add_argument("--submission", choices=[NATIVE, *SUBMISSION_TYPES], default=NATIVE,
                        help="submission type every model is exercised with")
    parser.add_argument("--stages", default="120:5,60:10,60:20,60:30,60:5",
                        help="<seconds>:<active users> stages; every user logs in during the first one")
    parser.add_argument("--spawn-rate", type=float, default=2.0, help="users (logins) per second")
    parser.add_argument("--processes", type=int, default=2, help="local Locust worker processes")
    parser.add_argument("--workers", type=int, default=0,
                        help="run as a master expecting this many remote Locust workers instead")
    parser.add_argument("--session-failure", type=float, default=0.1,
                        help="probability of a secure session failing on the seal timeout")
    parser.add_argument("--host", default=os.environ.get("BENCH_API_URL", "http://localhost:8000"))
    args = parser.parse_args()

    if "FED_AGG_INTERVAL_SECONDS" not in os.environ:
        raise SystemExit("FED_AGG_INTERVAL_SECONDS is not set, run with the stack's env file")
    interval = float(os.environ["FED_AGG_INTERVAL_SECONDS"])
    host = args.host.rstrip("/")
    stride = args.workers or args.processes
    stages = parse_stages(args.stages)
    users = max(active for _, active in stages)
    login_seconds = users / args.spawn_rate
    if stages[0][0] < login_seconds:
        raise SystemExit(f"the first stage lasts {stages[0][0]:g}s but {users} logins at "
                         f"{args.spawn_rate:g}/s take {login_seconds:g}s")
    models = select_models(host, args.split, args.submission, args.models)
    check_users(host, users, stride)

    label = "custom" if args.models else args.split
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{args.topology}-{label}-{args.submission}"
    run_dir = RESULTS_DIR / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists")
    run_dir.mkdir(parents=True)
    print(f"run {run_id}: {args.topology}, models " + ", ".join(f"{key} ({count})" for key, count in models.items()))

    schedule_start = time.time()
    boundaries = [schedule_start]
    for seconds, _ in stages:
        boundaries.append(boundaries[-1] + seconds)
    manifest = {
        "run_id": run_id,
        "topology": args.topology,
        "split": label,
        "submission_type": args.submission,
        "models": models,
        "load": {"users": users, "spawn_rate": args.spawn_rate, "stages": args.stages,
                 "processes": stride, "interval": interval,
                 "session_failure": args.session_failure, "host": host},
        "stages": [{"start": iso(begin), "end": iso(end), "active": active}
                   for (_, active), begin, end in zip(stages, boundaries, boundaries[1:])],
        "start": iso(schedule_start),
        "load_end": None,
        "end": None,
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    distribution = (["--master", "--expect-workers", str(args.workers)] if args.workers
                    else ["--processes", str(args.processes)])
    locust = subprocess.run([
        sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless", "--only-summary",
        *distribution, "--host", host, "--users", str(users), "--spawn-rate", str(args.spawn_rate),
        "--run-time", f"{math.ceil(boundaries[-1] - schedule_start)}s",
        "--run-dir", str(run_dir), "--models", ",".join(models), "--interval", str(interval),
        "--user-stride", str(stride), "--session-failure", str(args.session_failure),
        "--stages", args.stages, "--schedule-start", str(schedule_start),
    ])
    manifest["load_end"] = iso(time.time())
    manifest["locust_exit_code"] = locust.returncode
    write_manifest(manifest_path, manifest)
    print(f"load finished, collect run {run_id} before starting another")


if __name__ == "__main__":
    main()
