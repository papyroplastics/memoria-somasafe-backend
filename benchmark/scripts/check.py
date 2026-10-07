import argparse
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import func, select, text
from sqlmodel import Session

from common.celery_tasks import (
    CLEANUP_TASK,
    FED_AGG_TASK,
    FED_DISPATCH_TASK,
    QUANTIZE_TASK,
    SECURE_AGG_TASK,
    SECURE_SWEEP_TASK,
)
from common.config import (
    CLEANUP_INTERVAL_SECONDS,
    FED_AGG_INTERVAL_SECONDS,
    RESULTS_DIR,
    SECURE_SWEEP_INTERVAL_SECONDS,
)
from common.db import JobStatus, QuantizationJob, engine

PASS, FAIL, SKIP = "pass", "fail", "skip"
BALANCE_TOLERANCE = 0.05
STAGE_TOLERANCE = 0.05
RUNTIME_TOLERANCE = 0.10
BEAT_TASKS = {
    CLEANUP_TASK: ("CLEANUP_INTERVAL_SECONDS", CLEANUP_INTERVAL_SECONDS),
    FED_DISPATCH_TASK: ("FED_AGG_INTERVAL_SECONDS", FED_AGG_INTERVAL_SECONDS),
    SECURE_SWEEP_TASK: ("SECURE_SWEEP_INTERVAL_SECONDS", SECURE_SWEEP_INTERVAL_SECONDS),
}
HEAVY_TASKS = (FED_AGG_TASK, SECURE_AGG_TASK, QUANTIZE_TASK)
STAGES_AFTER = {"federated_aggregation": "cohort", "secure_aggregation": "members",
                "quantize_submission": "model_key"}
FINISHED = ("aggregated", "done")


class Run:
    def __init__(self, run_dir: Path, prometheus: str, env_file: Path):
        self.dir = run_dir
        self.config = dotenv_values(env_file)
        self.manifest = json.loads((run_dir / "manifest.json").read_text())
        self.prometheus = prometheus.rstrip("/")
        self.start = pd.Timestamp(self.manifest["start"]).timestamp()
        self.end = pd.Timestamp(self.manifest["end"]).timestamp()
        self.load_end = pd.Timestamp(self.manifest["load_end"]).timestamp()
        self.range = f"{round(self.end - self.start)}s"

    def query(self, query: str) -> list[tuple[dict, float]]:
        params = urllib.parse.urlencode({"query": query, "time": self.end})
        with urllib.request.urlopen(f"{self.prometheus}/api/v1/query?{params}", timeout=60) as resp:
            body = json.load(resp)
        if body["status"] != "success":
            raise RuntimeError(f"query failed: {query}: {body.get('error')}")
        return [(series["metric"], float(series["value"][1])) for series in body["data"]["result"]]

    def per(self, query: str, label: str) -> dict[str, float]:
        return {metric.get(label, ""): value for metric, value in self.query(query)}

    def interval(self, key: str, default: int) -> float:
        return float(self.config.get(key) or default)

    def worker_csv(self, task: str) -> pd.DataFrame | None:
        path = self.dir / f"worker_{task}.csv"
        return pd.read_csv(path) if path.exists() else None


def targets_up(run: Run) -> tuple[str, str]:
    down = [f"{metric.get('job')}/{metric.get('instance')}" for metric, value in
            run.query(f"min_over_time(up[{run.range}])") if value < 1]
    return (FAIL, "down at some point: " + ", ".join(down)) if down else (PASS, "every target up")


def task_failures(run: Run) -> tuple[str, str]:
    failed = {name: value for name, value in
              run.per(f"sum by (name) (increase(celery_task_failed_total[{run.range}]))", "name").items()
              if round(value) > 0}
    if failed:
        return FAIL, ", ".join(f"{name}: {value:.0f}" for name, value in failed.items())
    return PASS, "no failed tasks"


def beat_singleton(run: Run) -> tuple[str, str]:
    counts = run.per(f"sum by (name) (increase(celery_task_succeeded_total[{run.range}]))", "name")
    details, status = [], PASS
    for task, (key, default) in BEAT_TASKS.items():
        expected = (run.end - run.start) / run.interval(key, default)
        seen = counts.get(task, 0.0)
        if abs(seen - expected) > 1.5:
            status = FAIL
        details.append(f"{task.rsplit('.', 1)[-1]} {seen:.0f}/{expected:.1f}")
    return status, ", ".join(details)


def fan_out(run: Run) -> tuple[str, str]:
    if run.manifest["topology"] != "2x2":
        return SKIP, "single worker"
    names = "|".join(HEAVY_TASKS)
    hosts = run.per(f'sum by (hostname) (increase(celery_task_succeeded_total{{name=~"{names}"}}[{run.range}]))',
                    "hostname")
    busy = {host: value for host, value in hosts.items() if round(value) > 0}
    detail = ", ".join(f"{host}: {value:.0f}" for host, value in hosts.items()) or "no heavy tasks"
    return (PASS if len(busy) >= 2 else FAIL), detail


def load_balance(run: Run) -> tuple[str, str]:
    if run.manifest["topology"] != "2x2":
        return SKIP, "single gateway"
    counts = run.per(f"sum by (instance) (increase(http_requests_total[{run.range}]))", "instance")
    detail = ", ".join(f"{instance}: {value:.0f}" for instance, value in counts.items())
    if len(counts) < 2:
        return FAIL, detail or "no gateway requests"
    values = list(counts.values())
    spread = (max(values) - min(values)) / (sum(values) / len(values))
    return (PASS if spread <= BALANCE_TOLERANCE else FAIL), f"{detail} (spread {spread:.1%})"


def duplicate_rounds(run: Run) -> tuple[str, str]:
    with Session(engine) as session:
        duplicates = session.execute(text(
            "SELECT model_key, version_id, parent_weights_id, count(*) FROM globalweights "
            "WHERE parent_weights_id IS NOT NULL GROUP BY 1, 2, 3 HAVING count(*) > 1")).all()
        rounds = session.execute(text(
            "SELECT count(*) FROM globalweights WHERE parent_weights_id IS NOT NULL")).scalar_one()
    if duplicates:
        return FAIL, ", ".join(f"{key} v{version} parent {parent}: {count}" for key, version, parent, count in duplicates)
    return PASS, f"{rounds} aggregated snapshot(s), none share a parent"


def secure_timeouts(run: Run) -> tuple[str, str]:
    path = run.dir / "secure_rounds.csv"
    rounds = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if rounds.empty:
        return SKIP, "no secure rounds"
    unfinished = rounds[~rounds["status"].isin(["aggregated", "failed"])]
    created = pd.to_datetime(rounds["created_at"], utc=True, format="ISO8601")
    finished = pd.to_datetime(rounds["finished_at"], utc=True, format="ISO8601")
    cutoff = pd.Timestamp(run.load_end, unit="s", tz="UTC")
    orphaned = []
    for index, failed in rounds[(rounds["status"] == "failed") & (finished < cutoff)].iterrows():
        later = rounds[(rounds["model_key"] == failed["model_key"]) & (created > finished[index])]
        if later.empty:
            orphaned.append(str(failed["id"]))
    failures = (rounds["status"] == "failed").sum()
    detail = (f"{failures}/{len(rounds)} failed (expected ~{run.manifest['load']['round_failure']:.0%}), "
              f"{len(unfinished)} unfinished")
    if orphaned:
        detail += f", no fresh round after {', '.join(orphaned)}"
    return (FAIL if len(unfinished) or orphaned else PASS), detail


def pending_jobs(run: Run) -> tuple[str, str]:
    with Session(engine) as session:
        counts = dict(session.execute(
            select(QuantizationJob.status, func.count())
            .group_by(QuantizationJob.status)).all())  # type: ignore[arg-type]
    stuck = counts.get(JobStatus.pending, 0) + counts.get(JobStatus.running, 0)
    detail = ", ".join(f"{getattr(status, 'value', status)}: {count}" for status, count in counts.items())
    return (FAIL if stuck else PASS), detail or "no quantization jobs"


def stage_sums(run: Run) -> tuple[str, str]:
    details, status = [], PASS
    for task, last_field in STAGES_AFTER.items():
        frame = run.worker_csv(task)
        if frame is None or frame.empty:
            continue
        frame = frame[frame["outcome"].isin(FINISHED)]
        if frame.empty:
            continue
        stages = list(frame.columns[frame.columns.get_loc(last_field) + 1:])
        gap = ((frame["total_seconds"] - frame[stages].fillna(0).sum(axis=1)) / frame["total_seconds"]).abs()
        worst = gap.quantile(0.95)
        if worst > STAGE_TOLERANCE:
            status = FAIL
        details.append(f"{task} p95 gap {worst:.1%} over {len(frame)}")
    return (status, ", ".join(details)) if details else (SKIP, "no worker rows")


def runtime_agreement(run: Run) -> tuple[str, str]:
    exporter = run.per(f"sum by (name) (increase(celery_task_runtime_sum[{run.range}]))", "name")
    details, status = [], PASS
    for task in STAGES_AFTER:
        frame = run.worker_csv(task)
        reported = exporter.get(f"worker.tasks.{task}")
        if frame is None or frame.empty or not reported:
            continue
        ratio = frame["total_seconds"].sum() / reported
        if abs(ratio - 1) > RUNTIME_TOLERANCE:
            status = FAIL
        details.append(f"{task} csv/exporter {ratio:.2f}")
    return (status, ", ".join(details)) if details else (SKIP, "no comparable tasks")


def generator_cpu(run: Run) -> tuple[str, str]:
    return SKIP, "needs node-exporter on the client hosts"


CHECKS: dict[str, Callable[[Run], tuple[str, str]]] = {
    "targets_up": targets_up,
    "task_failures": task_failures,
    "beat_singleton": beat_singleton,
    "fan_out": fan_out,
    "load_balance": load_balance,
    "duplicate_rounds": duplicate_rounds,
    "secure_timeouts": secure_timeouts,
    "pending_jobs": pending_jobs,
    "stage_sums": stage_sums,
    "runtime_agreement": runtime_agreement,
    "generator_cpu": generator_cpu,
}


def run(run_dir: Path, prometheus: str, env_file: Path) -> bool:
    benchmark = Run(run_dir, prometheus, env_file)
    results = {}
    for name, check in CHECKS.items():
        try:
            status, detail = check(benchmark)
        except Exception as exc:
            status, detail = FAIL, f"check errored: {exc}"
        results[name] = {"status": status, "detail": detail}
        print(f"{status.upper():>4}  {name}: {detail}")
    if not benchmark.manifest.get("settled", True):
        results["settled"] = {"status": FAIL, "detail": "the stack did not settle after the load"}
        print(f"FAIL  settled: {results['settled']['detail']}")
    (run_dir / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    return all(result["status"] != FAIL for result in results.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pass/fail checks against a finished benchmark run")
    parser.add_argument("run_id")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--env-file", type=Path, default=Path("benchmark/prod.env"))
    args = parser.parse_args()
    raise SystemExit(0 if run(RESULTS_DIR / "benchmark" / args.run_id, args.prometheus, args.env_file) else 1)


if __name__ == "__main__":
    main()
