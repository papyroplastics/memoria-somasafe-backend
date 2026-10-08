"""Secure-aggregation endpoints (SubmissionType.secure). A client that trained on
the active weights joins that base's open session; once sealed, each member uploads
a masked vector and only the session's sum is ever unmasked. See
shared/docs/secure-aggregation.md.
"""

import base64

from fastapi import Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from common.config import (
    SECURE_CLIP_BOUND,
    SECURE_JOIN_COOLDOWN_SECONDS,
    SECURE_SESSION_MAX_MEMBERS,
    SUBMIT_DAILY_LIMIT,
    SUBMIT_DAILY_WINDOW_SECONDS,
)
from common.db import (
    GlobalWeights,
    ModelVersion,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    SubmissionType,
    get_latest_weights,
    get_open_session,
    get_session,
    utcnow,
)
from common.ratelimit import RateLimit, add_usage
from common.secure_agg import RING_MODULUS
from common.secure_session import seal_session
from api.lib.ratelimit import check_limit
from api.lib.session import get_current_user_id
from api.lib.challenge import require_device_owner
from .model import require_submission_type, router

_KA_KEY_LEN = 65

class SecureJoinRequest(BaseModel):
    ka_public_key: str


class SecureJoinResponse(BaseModel):
    session_id: int
    base_weights_id: int
    user_id: int


class RosterEntry(BaseModel):
    user_id: int
    ka_public_key: str


class SecureSessionDescriptor(BaseModel):
    session_id: int
    model_key: str
    base_weights_id: int
    weight_count: int
    member_count: int
    clip_bound: float
    scale: int
    ring_modulus: int
    roster: list[RosterEntry]


def _decode_ka_key(encoded: str) -> bytes:
    try:
        key = base64.b64decode(encoded, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="ka_public_key is not valid base64")
    if len(key) != _KA_KEY_LEN or key[0] != 0x04:
        raise HTTPException(status_code=400,
                            detail="ka_public_key must be a 65-byte uncompressed P-256 point")
    return key


def _active_base(session: Session, key: str, weights_id: int) -> GlobalWeights:
    base = session.get(GlobalWeights, weights_id)
    if base is None or base.model_key != key:
        raise HTTPException(status_code=400,
                            detail=f"Unknown base weights for model '{key}'")
    active = get_latest_weights(session, key)
    if active is None or active.id != base.id:
        raise HTTPException(
            status_code=409,
            detail=f"Stale base weights; re-download the latest weights of "
                   f"'{key}' before joining")
    return base


def _open_session(session: Session, base: GlobalWeights) -> SecureSession:
    current = get_open_session(session, base.id, lock=True)
    for _ in range(3):
        if current is not None:
            return current
        session.execute(
            pg_insert(SecureSession)
            .values(model_key=base.model_key, version_id=base.version_id,
                    base_weights_id=base.id, clip_bound=SECURE_CLIP_BOUND,
                    status=SecureSessionStatus.open, created_at=utcnow())
            .on_conflict_do_nothing(index_elements=["base_weights_id"],
                                    index_where=text("status = 'open'")))
        current = get_open_session(session, base.id, lock=True)
    raise HTTPException(status_code=503, detail="Could not open a session; retry")


@router.post("/secure/join/{key}/{weights_id}", response_model=SecureJoinResponse,
             status_code=202)
def secure_join(key: str, weights_id: int, body: SecureJoinRequest,
                session: Session = Depends(get_session),
                user_id: int = Depends(get_current_user_id)):
    check_limit(RateLimit.secure_join, user_id, key, 1, SECURE_JOIN_COOLDOWN_SECONDS)
    require_submission_type(session, key, {SubmissionType.secure})
    require_device_owner(session, user_id)
    ka_key = _decode_ka_key(body.ka_public_key)

    try:
        base = _active_base(session, key, weights_id)
        seat = session.exec(
            select(SecureSessionMember)
            .where(SecureSessionMember.base_weights_id == base.id,
                   SecureSessionMember.user_id == user_id)).first()
        if seat is None:
            current = _open_session(session, base)
            seat = SecureSessionMember(session_id=current.id, user_id=user_id,
                                       base_weights_id=base.id, ka_public_key=ka_key)
        else:
            current = session.get(SecureSession, seat.session_id, with_for_update=True,
                                  populate_existing=True)
            if current.status != SecureSessionStatus.open:
                raise HTTPException(status_code=409,
                                    detail="Already participating on these weights")
            seat.ka_public_key = ka_key
        session.add(seat)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status_code=409,
                                detail="Already participating on these weights")

        members = session.exec(select(func.count()).select_from(SecureSessionMember)
                               .where(SecureSessionMember.session_id == current.id)).one()
        if members >= SECURE_SESSION_MAX_MEMBERS:
            seal_session(session, current.id)
        session.commit()
        return SecureJoinResponse(session_id=current.id, base_weights_id=base.id,
                                  user_id=user_id)
    finally:
        add_usage(RateLimit.secure_join, user_id, key, SECURE_JOIN_COOLDOWN_SECONDS)


def _require_member(session: Session, session_id: int,
                    user_id: int) -> tuple[SecureSession, SecureSessionMember]:
    current = session.get(SecureSession, session_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Session not found")
    member = session.get(SecureSessionMember, (session_id, user_id))
    if member is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return current, member


@router.get("/secure/session/{session_id}", response_model=SecureSessionDescriptor)
def secure_descriptor(session_id: int,
                      session: Session = Depends(get_session),
                      user_id: int = Depends(get_current_user_id)):
    current, _ = _require_member(session, session_id, user_id)
    if current.status != SecureSessionStatus.sealed:
        raise HTTPException(status_code=409,
                            detail=f"Session is {current.status.value}, not sealed")

    members = session.exec(
        select(SecureSessionMember)
        .where(SecureSessionMember.session_id == session_id)
        .order_by(SecureSessionMember.user_id.asc())  # type: ignore[attr-defined]
    ).all()
    version = session.get(ModelVersion, current.version_id)
    return SecureSessionDescriptor(
        session_id=current.id, model_key=current.model_key,
        base_weights_id=current.base_weights_id, weight_count=version.weight_count,
        member_count=current.member_count, clip_bound=current.clip_bound,
        scale=current.scale, ring_modulus=RING_MODULUS,
        roster=[RosterEntry(user_id=m.user_id,
                            ka_public_key=base64.b64encode(m.ka_public_key).decode())
                for m in members],
    )


@router.post("/secure/submit/{session_id}", status_code=202)
def secure_submit(session_id: int, body: bytes = Body(...),
                  session: Session = Depends(get_session),
                  user_id: int = Depends(get_current_user_id)):
    current, member = _require_member(session, session_id, user_id)
    require_device_owner(session, user_id)
    check_limit(RateLimit.weight_submit, user_id, current.model_key,
                SUBMIT_DAILY_LIMIT, SUBMIT_DAILY_WINDOW_SECONDS)

    if current.status != SecureSessionStatus.sealed:
        raise HTTPException(
            status_code=409,
            detail=f"Session is {current.status.value}, not accepting submissions")
    if member.masked is not None:
        raise HTTPException(status_code=409, detail="Already submitted for this session")

    active = get_latest_weights(session, current.model_key)
    if active is None or active.id != current.base_weights_id:
        raise HTTPException(status_code=409,
                            detail="Session base weights are stale; the session is void")

    version = session.get(ModelVersion, current.version_id)
    if len(body) != version.weight_count * 4:
        raise HTTPException(status_code=400,
                            detail=f"Expected {version.weight_count} little-endian uint32 elements")

    try:
        member.masked = bytes(body)
        member.submitted_at = utcnow()
        session.add(member)
        session.commit()
        return {"session_id": session_id, "submitted": True}
    finally:
        add_usage(RateLimit.weight_submit, user_id, current.model_key,
                  SUBMIT_DAILY_WINDOW_SECONDS)
