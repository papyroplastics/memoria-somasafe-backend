import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import redis
from sqlalchemy import func, select, text, update
from sqlmodel import Session

from common.celery_tasks import CLEANUP_TASK, FED_DISPATCH_TASK, HEAVY_QUEUE, LIGHT_QUEUE
from common.config import BROKER_URL, REDIS_URL, RESULTS_DIR
from common.db import (
    ClientDeltaSubmission,
    GlobalWeights,
    JobStatus,
    QuantizationJob,
    SecurePartial,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    engine,
    get_active_weights_ids,
    get_latest_version,
    utcnow,
)
from worker.celery_app import app

SECURE_SESSION_COLUMNS = ("id", "model_key", "status", "base_weights_id", "member_count",
                          "created_at", "sealed_at", "summing_at", "finished_at")
UNFINISHED_JOBS = (JobStatus.pending, JobStatus.running)
UNFINISHED_SESSIONS = (SecureSessionStatus.sealed, SecureSessionStatus.summing)
SETTLE_POLL_SECONDS = 5
SETTLE_TIMEOUT_SECONDS = 600
TRUNCATED = ("quantizationresult", "quantizationjob", "clientdeltasubmission",
             "securepartial", "securesessionmember", "securesession", "authsession")
KOMBU_BINDINGS = b"_kombu.binding."
PASS, FAIL, SKIP = "pass", "fail", "skip"


def drained(broker: redis.Redis) -> bool:
    return not sum(broker.llen(queue) for queue in (LIGHT_QUEUE, HEAVY_QUEUE)) + broker.hlen("unacked")


def quiet(broker: redis.Redis) -> bool:
    abandon_sessions()
    if not drained(broker):
        return False
    with Session(engine) as session:
        jobs = session.execute(select(func.count()).select_from(QuantizationJob)
                               .where(QuantizationJob.status.in_(UNFINISHED_JOBS))).scalar_one()  # type: ignore[attr-defined]
        sessions = session.execute(select(func.count()).select_from(SecureSession)
                                   .where(SecureSession.status.in_(UNFINISHED_SESSIONS))).scalar_one()  # type: ignore[attr-defined]
    return jobs == 0 and sessions == 0


def aggregated(broker: redis.Redis) -> bool:
    if not drained(broker):
        return False
    with Session(engine) as session:
        partials = session.execute(
            select(func.count()).select_from(SecurePartial)
            .join(SecureSession, SecureSession.id == SecurePartial.session_id)  # type: ignore[arg-type]
            .where(~select(GlobalWeights.id)
                   .where(GlobalWeights.parent_weights_id == SecureSession.base_weights_id,
                          GlobalWeights.valid == True)
                   .exists())).scalar_one()
    return partials == 0


def abandon_sessions() -> int:
    submitted = (select(func.count()).select_from(SecureSessionMember)
                 .where(SecureSessionMember.session_id == SecureSession.id,
                        SecureSessionMember.submitted_at.is_not(None))  # type: ignore[union-attr]
                 .scalar_subquery())
    with Session(engine) as session:
        count = session.execute(
            update(SecureSession)
            .where(SecureSession.status == SecureSessionStatus.sealed,  # type: ignore[arg-type]
                   submitted < SecureSession.member_count)
            .values(status=SecureSessionStatus.failed, finished_at=utcnow())).rowcount
        session.commit()
    return count


def wait(done, label: str) -> bool:
    print(f"waiting for {label}")
    deadline = time.time() + SETTLE_TIMEOUT_SECONDS
    while not done():
        if time.time() >= deadline:
            print(f"timed out waiting for {label}")
            return False
        time.sleep(SETTLE_POLL_SECONDS)
    return True


def settle() -> bool:
    broker = redis.from_url(BROKER_URL)
    if not wait(lambda: quiet(broker), "the queues and secure sessions to settle"):
        return False
    app.send_task(FED_DISPATCH_TASK)
    if not wait(lambda: aggregated(broker), "the last round of every model"):
        return False
    app.send_task(CLEANUP_TASK)
    return wait(lambda: drained(broker), "the cleanup sweep")


def round_limits(models: dict) -> dict[str, int]:
    with Session(engine) as session:
        return {key: version.round_limit for key in models
                if (version := get_latest_version(session, key)) is not None}


def save_secure_sessions(path: Path) -> pd.DataFrame:
    columns = [getattr(SecureSession, column) for column in SECURE_SESSION_COLUMNS]
    with Session(engine) as session:
        rows = session.execute(select(*columns).order_by(SecureSession.id)).all()  # type: ignore[arg-type]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SECURE_SESSION_COLUMNS)
        for row in rows:
            writer.writerow([value.isoformat() if isinstance(value, datetime)
                             else getattr(value, "value", value) for value in row])
    return pd.read_csv(path)


def duplicate_rounds(manifest: dict, sessions: pd.DataFrame) -> tuple[str, str]:
    with Session(engine) as session:
        duplicates = session.execute(text(
            "SELECT model_key, version_id, parent_weights_id, count(*) FROM globalweights "
            "WHERE parent_weights_id IS NOT NULL GROUP BY 1, 2, 3 HAVING count(*) > 1")).all()
        rounds = session.execute(text(
            "SELECT count(*) FROM globalweights WHERE parent_weights_id IS NOT NULL")).scalar_one()
    if duplicates:
        return FAIL, ", ".join(f"{key} v{version} parent {parent}: {count}" for key, version, parent, count in duplicates)
    return PASS, f"{rounds} aggregated snapshot(s), none share a parent"


def secure_sessions(manifest: dict, sessions: pd.DataFrame) -> tuple[str, str]:
    if sessions.empty:
        return SKIP, "no secure sessions"
    unfinished = sessions[sessions["status"].isin(["sealed", "summing"])]
    left_open = (sessions["status"] == "open").sum()
    created = pd.to_datetime(sessions["created_at"], utc=True, format="ISO8601")
    finished = pd.to_datetime(sessions["finished_at"], utc=True, format="ISO8601")
    cutoff = pd.Timestamp(manifest["load_end"]) - pd.Timedelta(seconds=manifest["load"]["period"])
    failed = sessions["status"] == "failed"
    orphaned = []
    for index, session in sessions[failed & (finished < cutoff)].iterrows():
        later = sessions[(sessions["model_key"] == session["model_key"]) & (created > finished[index])]
        if later.empty:
            orphaned.append(str(session["id"]))
    with Session(engine) as session:
        replaced = dict(session.execute(text(
            "SELECT parent_weights_id, min(created_at) FROM globalweights "
            "WHERE parent_weights_id IS NOT NULL GROUP BY 1")).all())
    replaced_at = pd.to_datetime(sessions["base_weights_id"].map(replaced), utc=True)
    sealed = sessions["sealed_at"].notna()
    abandoned = failed & sealed & (finished >= pd.Timestamp(manifest["load_end"]))
    counted = sealed & ~abandoned
    stale = failed & counted & (replaced_at <= finished)
    closed = failed & sessions["summing_at"].notna()
    detail = (f"{(sessions['status'] == 'summed').sum()}/{len(sessions)} summed, "
              f"{(failed & ~sealed).sum()} failed open, {(failed & counted).sum()}/{counted.sum()} failed sealed "
              f"(expected ~{manifest['load']['session_failure']:.0%}, {stale.sum()} on replaced weights, "
              f"{closed.sum()} summed after their round closed), "
              f"{abandoned.sum()} failed after the load ended, {left_open} left open, {len(unfinished)} unfinished")
    if orphaned:
        detail += f", no fresh session after {', '.join(orphaned)}"
    return (FAIL if len(unfinished) or orphaned else PASS), detail


def secure_partials(manifest: dict, sessions: pd.DataFrame) -> tuple[str, str]:
    with Session(engine) as session:
        left = session.execute(select(func.count()).select_from(SecurePartial)).scalar_one()
    return (FAIL, f"{left} partial result(s) never aggregated") if left else (PASS, "every partial aggregated")


def pending_jobs(manifest: dict, sessions: pd.DataFrame) -> tuple[str, str]:
    with Session(engine) as session:
        counts = dict(session.execute(
            select(QuantizationJob.status, func.count())
            .group_by(QuantizationJob.status)).all())  # type: ignore[arg-type]
    stuck = counts.get(JobStatus.pending, 0) + counts.get(JobStatus.running, 0)
    detail = ", ".join(f"{getattr(status, 'value', status)}: {count}" for status, count in counts.items())
    return (FAIL if stuck else PASS), detail or "no quantization jobs"


def round_sizes(manifest: dict, sessions: pd.DataFrame) -> tuple[str, str]:
    with Session(engine) as session:
        rows = session.execute(text(
            "SELECT w.model_key, v.round_limit, "
            "(SELECT count(*) FROM clientdeltasubmission d "
            " WHERE d.base_weights_id = w.parent_weights_id AND d.valid) + "
            "(SELECT coalesce(sum(s.member_count), 0) FROM securesession s "
            " WHERE s.base_weights_id = w.parent_weights_id AND s.status = 'summed') "
            "FROM globalweights w JOIN modelversion v ON v.id = w.version_id "
            "WHERE w.parent_weights_id IS NOT NULL AND w.valid ORDER BY w.id")).all()
    if not rows:
        return SKIP, "no aggregated rounds"
    rounds: dict[str, list[tuple[int, int]]] = {}
    for key, limit, submissions in rows:
        rounds.setdefault(key, []).append((limit, submissions))
    parts, early = [], 0
    for key, sizes in rounds.items():
        full = [submissions - limit for limit, submissions in sizes if submissions >= limit]
        below = sum(submissions < limit for limit, submissions in sizes[:-1])
        early += below
        overshoot = f" (+{min(full)}..+{max(full)})" if full else ""
        parts.append(f"{key}: {len(sizes)} round(s) at limit {sizes[-1][0]}, {len(full)} full{overshoot}, "
                     f"{below} closed early, last {sizes[-1][1]}")
    return (FAIL if early else PASS), "; ".join(parts)


def payloads_cleared(manifest: dict, sessions: pd.DataFrame) -> tuple[str, str]:
    with Session(engine) as session:
        active = get_active_weights_ids(session)
        deltas = session.execute(
            select(func.count()).select_from(ClientDeltaSubmission)
            .where(ClientDeltaSubmission.deltas.is_not(None),  # type: ignore[union-attr]
                   ClientDeltaSubmission.base_weights_id.not_in(active))).scalar_one()  # type: ignore[attr-defined]
        masked = session.execute(
            select(func.count()).select_from(SecureSessionMember)
            .where(SecureSessionMember.masked.is_not(None),  # type: ignore[union-attr]
                   SecureSessionMember.base_weights_id.not_in(active))).scalar_one()  # type: ignore[attr-defined]
        partials = session.execute(
            select(func.count()).select_from(SecurePartial)
            .join(SecureSession, SecureSession.id == SecurePartial.session_id)  # type: ignore[arg-type]
            .where(SecureSession.base_weights_id.not_in(active))).scalar_one()  # type: ignore[attr-defined]
    detail = f"{deltas} delta(s), {masked} masked vector(s), {partials} partial(s) left on inactive weights"
    return (FAIL if deltas or masked or partials else PASS), detail


CHECKS = {
    "round_sizes": round_sizes,
    "payloads_cleared": payloads_cleared,
    "duplicate_rounds": duplicate_rounds,
    "secure_sessions": secure_sessions,
    "secure_partials": secure_partials,
    "pending_jobs": pending_jobs,
}


def reset() -> None:
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TRUNCATED)} RESTART IDENTITY"))
        deleted = conn.execute(text(
            "DELETE FROM globalweights WHERE id NOT IN "
            "(SELECT min(id) FROM globalweights GROUP BY version_id)")).rowcount
        conn.execute(text("UPDATE globalweights SET closed = false"))
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('globalweights', 'id'), "
            "(SELECT coalesce(max(id), 0) + 1 FROM globalweights), false)"))
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM ANALYZE globalweights, weightsartifact"))
    print(f"truncated {', '.join(TRUNCATED)}; dropped {deleted} aggregated weight snapshot(s)")

    store = redis.from_url(REDIS_URL)
    store_keys = store.dbsize()
    store.flushdb()
    print(f"flushed {store_keys} key(s) from {REDIS_URL}")

    broker = redis.from_url(BROKER_URL)
    keys = [key for key in broker.scan_iter(count=1000) if not key.startswith(KOMBU_BINDINGS)]
    for start in range(0, len(keys), 1000):
        broker.delete(*keys[start:start + 1000])
    print(f"dropped {len(keys)} key(s) from {BROKER_URL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Settle a finished run, save and check the stack's state, then reset it")
    parser.add_argument("run_id")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / "benchmark" / args.run_id
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["round_limits"] = round_limits(manifest["models"])
    if manifest["load_end"] is not None:
        print(f"failed {abandon_sessions()} sealed session(s) still missing submissions")
        manifest["settled"] = settle()
    else:
        manifest["settled"] = False
    manifest["end"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    sessions = save_secure_sessions(run_dir / "secure_sessions.csv")
    results = {"settled": {"status": PASS if manifest["settled"] else FAIL,
                           "detail": "settled" if manifest["settled"] else "did not settle after the load"}}
    for name, check in CHECKS.items():
        try:
            status, detail = check(manifest, sessions)
        except Exception as exc:
            status, detail = FAIL, f"check errored: {exc}"
        results[name] = {"status": status, "detail": detail}
    for name, result in results.items():
        print(f"{result['status'].upper():>4}  {name}: {result['detail']}")
    (run_dir / "checks.json").write_text(json.dumps(results, indent=2) + "\n")

    reset()


if __name__ == "__main__":
    main()
