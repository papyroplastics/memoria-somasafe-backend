"""Secure-aggregation endpoints (SubmissionType.secure). A client joins the open
round for a model, the cohort is sealed (roster + public keys frozen), then each
member uploads a *masked* weight vector; only the summed round is ever unmasked,
so the server sees the aggregate and never an individual update. The full scheme
and its invariants are in shared/docs/secure-aggregation.md.
"""

import base64

from fastapi import Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import Session, select

from common.config import (
    SECURE_CLIP_BOUND,
    SECURE_JOIN_COOLDOWN_SECONDS,
    SUBMIT_DAILY_LIMIT,
    SUBMIT_DAILY_WINDOW_SECONDS,
)
from common.db import (
    ModelVersion,
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    SubmissionType,
    get_latest_version,
    get_latest_weights,
    get_open_round,
    get_session,
    utcnow,
)
from common.ratelimit import RateLimit
from common.secure_agg import RING_MODULUS
from api.lib.ratelimit import check_limit, record_usage
from api.lib.session import get_current_user_id
from api.lib.challenge import require_device_owner
from .model import require_submission_type, router

_KA_KEY_LEN = 65

class SecureJoinRequest(BaseModel):
    ka_public_key: str


class SecureJoinResponse(BaseModel):
    round_id: int
    base_weights_id: int
    user_id: int


class RosterEntry(BaseModel):
    user_id: int
    ka_public_key: str


class SecureRoundDescriptor(BaseModel):
    round_id: int
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


def _open_round(session: Session, key: str, version_id: int,
                base_weights_id: int) -> SecureRound:
    round = get_open_round(session, key, lock=True)
    for _ in range(3):
        if round is not None:
            return round
        session.execute(
            pg_insert(SecureRound)
            .values(model_key=key, version_id=version_id, base_weights_id=base_weights_id,
                    clip_bound=SECURE_CLIP_BOUND, status=SecureRoundStatus.open,
                    created_at=utcnow())
            .on_conflict_do_nothing(index_elements=["model_key"],
                                    index_where=text("status = 'open'")))
        round = get_open_round(session, key, lock=True)
    raise HTTPException(status_code=503, detail="Could not open a round; retry")


@router.post("/secure/join/{key}", response_model=SecureJoinResponse, status_code=202)
def secure_join(key: str, body: SecureJoinRequest,
                session: Session = Depends(get_session),
                user_id: int = Depends(get_current_user_id)):
    check_limit(RateLimit.secure_join, user_id, key, 1, SECURE_JOIN_COOLDOWN_SECONDS)
    require_submission_type(session, key, {SubmissionType.secure})
    require_device_owner(session, user_id)
    ka_key = _decode_ka_key(body.ka_public_key)

    try:
        active = get_latest_weights(session, key)
        if active is None:
            raise HTTPException(status_code=404, detail=f"No weights for model '{key}'")

        round = _open_round(session, key, get_latest_version(session, key).id, active.id)
        if round.base_weights_id != active.id:
            # Stale open rounds are failed by the worker's sweep.
            raise HTTPException(status_code=409,
                                detail="Open round's base weights are stale; retry")

        member = session.get(SecureRoundMember, (round.id, user_id))
        if member is None:
            member = SecureRoundMember(round_id=round.id, user_id=user_id,
                                       ka_public_key=ka_key)
        else:
            member.ka_public_key = ka_key
        session.add(member)
        session.commit()
        return SecureJoinResponse(round_id=round.id, base_weights_id=active.id,
                                  user_id=user_id)
    finally:
        record_usage(RateLimit.secure_join, user_id, key, SECURE_JOIN_COOLDOWN_SECONDS)


def _require_member(session: Session, round_id: int,
                    user_id: int) -> tuple[SecureRound, SecureRoundMember]:
    round = session.get(SecureRound, round_id)
    if round is None:
        raise HTTPException(status_code=404, detail="Round not found")
    member = session.get(SecureRoundMember, (round_id, user_id))
    if member is None:
        raise HTTPException(status_code=404, detail="Round not found")
    return round, member


@router.get("/secure/round/{round_id}", response_model=SecureRoundDescriptor)
def secure_descriptor(round_id: int,
                      session: Session = Depends(get_session),
                      user_id: int = Depends(get_current_user_id)):
    round, _ = _require_member(session, round_id, user_id)
    if round.status != SecureRoundStatus.sealed:
        raise HTTPException(status_code=409,
                            detail=f"Round is {round.status.value}, not sealed")

    members = session.exec(
        select(SecureRoundMember)
        .where(SecureRoundMember.round_id == round_id)
        .order_by(SecureRoundMember.user_id.asc())  # type: ignore[attr-defined]
    ).all()
    version = session.get(ModelVersion, round.version_id)
    return SecureRoundDescriptor(
        round_id=round.id, model_key=round.model_key,
        base_weights_id=round.base_weights_id, weight_count=version.weight_count,
        member_count=round.member_count, clip_bound=round.clip_bound,
        scale=round.scale, ring_modulus=RING_MODULUS,
        roster=[RosterEntry(user_id=m.user_id,
                            ka_public_key=base64.b64encode(m.ka_public_key).decode())
                for m in members],
    )


@router.post("/secure/submit/{round_id}", status_code=202)
def secure_submit(round_id: int, body: bytes = Body(...),
                  session: Session = Depends(get_session),
                  user_id: int = Depends(get_current_user_id)):
    round, member = _require_member(session, round_id, user_id)
    require_device_owner(session, user_id)
    check_limit(RateLimit.weight_submit, user_id, round.model_key,
                SUBMIT_DAILY_LIMIT, SUBMIT_DAILY_WINDOW_SECONDS)

    if round.status != SecureRoundStatus.sealed:
        raise HTTPException(status_code=409,
                            detail=f"Round is {round.status.value}, not accepting submissions")
    if member.masked is not None:
        raise HTTPException(status_code=409, detail="Already submitted for this round")

    active = get_latest_weights(session, round.model_key)
    if active is None or active.id != round.base_weights_id:
        raise HTTPException(status_code=409,
                            detail="Round base weights are stale; the round is void")

    version = session.get(ModelVersion, round.version_id)
    if len(body) != version.weight_count * 4:
        raise HTTPException(status_code=400,
                            detail=f"Expected {version.weight_count} little-endian uint32 elements")

    try:
        member.masked = bytes(body)
        member.submitted_at = utcnow()
        session.add(member)
        session.commit()
        return {"round_id": round_id, "submitted": True}
    finally:
        record_usage(RateLimit.weight_submit, user_id, round.model_key,
                     SUBMIT_DAILY_WINDOW_SECONDS)
