from datetime import timedelta

from sqlalchemy import and_, delete, or_, select, update
from sqlmodel import Session

from common.celery_tasks import CLEANUP_TASK
from common.config import (
    CLEANUP_BATCH_SIZE,
    RESULT_TTL_SECONDS,
    SERVE_GRACE_SECONDS,
    WORKER_REAP_AFTER_SECONDS,
)
from common.db import JobStatus, QuantizationJob, QuantizationResult, engine, utcnow
from worker.celery_app import app


def _expire_batch(session: Session, grace_cutoff, ttl_cutoff) -> int:
    ids = list(session.execute(
        select(QuantizationJob.id)
        .join(QuantizationResult, QuantizationResult.job_id == QuantizationJob.id)  # type: ignore[arg-type]
        .where(or_(
            and_(QuantizationJob.served_at.is_not(None),  # type: ignore[union-attr]
                 QuantizationJob.served_at < grace_cutoff),  # type: ignore[operator]
            QuantizationJob.created_at < ttl_cutoff,
        ))
        .limit(CLEANUP_BATCH_SIZE)
        .with_for_update(of=QuantizationJob, skip_locked=True)  # type: ignore[arg-type]
    ).scalars())
    if ids:
        session.execute(delete(QuantizationResult)
                        .where(QuantizationResult.job_id.in_(ids)))  # type: ignore[attr-defined]
        session.execute(update(QuantizationJob)
                        .where(QuantizationJob.id.in_(ids))  # type: ignore[attr-defined]
                        .values(status=JobStatus.expired, signature=None))
    session.commit()
    return len(ids)


@app.task(name=CLEANUP_TASK)
def cleanup_results() -> dict[str, int]:
    now = utcnow()
    grace_cutoff = now - timedelta(seconds=SERVE_GRACE_SECONDS)
    ttl_cutoff = now - timedelta(seconds=RESULT_TTL_SECONDS)
    reap_cutoff = now - timedelta(seconds=WORKER_REAP_AFTER_SECONDS)

    expired = 0
    while True:
        with Session(engine) as session:
            batch = _expire_batch(session, grace_cutoff, ttl_cutoff)
        expired += batch
        if batch < CLEANUP_BATCH_SIZE:
            break

    with Session(engine) as session:
        lost = session.execute(
            update(QuantizationJob)
            .where(QuantizationJob.status == JobStatus.running,
                   QuantizationJob.started_at < reap_cutoff)  # type: ignore[operator]
            .values(status=JobStatus.failed, error="worker lost", finished_at=now)
        ).rowcount  # type: ignore[attr-defined]
        abandoned = session.execute(
            update(QuantizationJob)
            .where(QuantizationJob.status == JobStatus.pending,
                   QuantizationJob.created_at < ttl_cutoff)
            .values(status=JobStatus.expired)
        ).rowcount  # type: ignore[attr-defined]
        session.commit()

    return {"expired": expired, "lost": lost, "abandoned": abandoned}
