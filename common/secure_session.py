from sqlalchemy import func
from sqlmodel import Session, select

from common.db import SecureSession, SecureSessionMember, SecureSessionStatus, utcnow
from common.secure_agg import compute_scale


def seal_session(session: Session, session_id: int) -> int | None:
    """Freeze an open session's roster and fix n and S = floor(2^31/(n*B)).
    ``None`` if it was no longer open. The caller commits."""
    if not SecureSession.transition(session, session_id, SecureSessionStatus.open,
                                    SecureSessionStatus.sealed, sealed_at=utcnow()):
        return None
    n = session.exec(select(func.count()).select_from(SecureSessionMember)
                     .where(SecureSessionMember.session_id == session_id)).one()
    row = session.get(SecureSession, session_id, populate_existing=True)
    row.member_count = n
    row.scale = compute_scale(n, row.clip_bound)
    session.add(row)
    return n
