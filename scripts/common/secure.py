"""DB-side helpers for the secure-aggregation harnesses (no HTTP). Sealing a round
plays the operator a real deployment would automate — there is deliberately no seal
endpoint, so a client can never freeze a roster mid-join.
"""

from sqlmodel import Session, select

from common.db import (
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    engine,
    utcnow,
)
from common.secure_agg import compute_scale


def seal_round(round_id: int, min_members: int) -> int:
    """Freeze the roster and fix n and the scale S = floor(2^31/(n*B))."""
    with Session(engine) as session:
        round = session.get(SecureRound, round_id)
        if round is None:
            raise SystemExit(f"round {round_id} vanished before seal")
        members = list(session.exec(
            select(SecureRoundMember).where(SecureRoundMember.round_id == round_id)))
        n = len(members)
        if n < min_members:
            raise SystemExit(f"only {n} members joined, need >= {min_members} to seal")
        round.member_count = n
        round.scale = compute_scale(n, round.clip_bound)
        round.status = SecureRoundStatus.sealed
        round.sealed_at = utcnow()
        session.add(round)
        session.commit()
        return n
