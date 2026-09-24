"""DB-side helpers for the secure-aggregation harnesses (no HTTP). The worker's sweep
seals and dispatches rounds on its own; the harnesses do both by hand for determinism,
tolerating the sweep having got there first.
"""

import time

from sqlalchemy import func
from sqlmodel import Session, select

from common.celery_tasks import SECURE_AGG_TASK
from common.db import (
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    engine,
)
from common.secure_round import seal_round as seal


def seal_round(round_id: int, min_members: int) -> int:
    with Session(engine) as session:
        n = session.exec(select(func.count()).select_from(SecureRoundMember)
                         .where(SecureRoundMember.round_id == round_id)).one()
        if n < min_members:
            raise SystemExit(f"only {n} members joined, need >= {min_members} to seal")
        sealed = seal(session, round_id)
        session.commit()
        if sealed is not None:
            return sealed
        round = session.get(SecureRound, round_id, populate_existing=True)
        if round is None or round.status is not SecureRoundStatus.sealed:
            raise SystemExit(f"round {round_id} could not be sealed "
                             f"({round.status.value if round else 'missing'})")
        return round.member_count


def run_round(app, round_id: int, timeout: float = 300.0) -> str:
    """Dispatch the round's aggregation and wait for the round row to settle; the
    sweep may have dispatched it already, so the task's own result is not used."""
    app.send_task(SECURE_AGG_TASK, args=[round_id])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with Session(engine) as session:
            round = session.get(SecureRound, round_id)
            if round.status is SecureRoundStatus.aggregated:
                return f"aggregated {round.member_count} members into new global weights"
            if round.status is SecureRoundStatus.failed:
                raise SystemExit(f"secure round produced no new weights: {round.error}")
        time.sleep(1.0)
    raise SystemExit(f"secure round {round_id} did not settle within {timeout:.0f}s")
