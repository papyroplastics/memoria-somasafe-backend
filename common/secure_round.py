from sqlalchemy import func
from sqlmodel import Session, select

from common.db import SecureRound, SecureRoundMember, SecureRoundStatus, utcnow
from common.secure_agg import compute_scale


def seal_round(session: Session, round_id: int) -> int | None:
    """Freeze an open round's roster and fix n and the scale S = floor(2^31/(n*B)).
    ``None`` if the round was no longer open. The caller commits."""
    if not SecureRound.transition(session, round_id, SecureRoundStatus.open,
                                  SecureRoundStatus.sealed, sealed_at=utcnow()):
        return None
    n = session.exec(select(func.count()).select_from(SecureRoundMember)
                     .where(SecureRoundMember.round_id == round_id)).one()
    round = session.get(SecureRound, round_id, populate_existing=True)
    round.member_count = n
    round.scale = compute_scale(n, round.clip_bound)
    session.add(round)
    return n
