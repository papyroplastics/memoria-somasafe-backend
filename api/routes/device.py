"""Device ownership attestation.

A client proves control of a physical device by having it sign a server-issued
challenge with its factory ECDSA P-256 key:

1. ``POST /device/challenge`` {serial} -> a random nonce + challenge metadata.
2. The client builds the canonical payload (below), has the device sign its
   SHA-256, and submits the DER signature to ``POST /device/attest``.
3. The server rebuilds the same payload, verifies the signature against the
   device's stored public key, and on success records the user as the owner.

The canonical payload is a fixed concatenation (big-endian, no separators):

    nonce(32B) || instance_id(16B uuid) || server_time(u64) || user_id(u64) || serial(ascii)

A device's owner can only change once per DEVICE_ATTEST_COOLDOWN_SECONDS; only a
successful attestation consumes that window (failures/timeouts do not).
"""

import base64
import secrets
import uuid
from datetime import timedelta

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session, select

from api.lib import challenge
from common.config import DEVICE_ATTEST_COOLDOWN_SECONDS
from common.db import Device, User, get_session, utcnow
from .auth import get_current_user

router = APIRouter(prefix="/device")

NONCE_LENGTH = 32


class ChallengeRequest(BaseModel):
    serial: str


class ChallengeResponse(BaseModel):
    instance_id: str
    nonce: str          # base64 (standard) of the raw nonce
    server_time: int    # epoch seconds
    user_id: int


class AttestRequest(BaseModel):
    instance_id: str
    signature: str      # base64 (standard) of the DER ECDSA signature


@router.get("/owned")
def owned_devices(session: Session = Depends(get_session),
                  user: User = Depends(get_current_user)) -> list[str]:
    """Serials of every device the caller currently owns. Lets a client check
    whether the server still considers it the owner (ownership can be lost when
    someone else re-attests the same device)."""
    return list(session.exec(
        select(Device.serial).where(Device.owner_id == user.id)
    ).all())


@router.post("/challenge")
def request_challenge(body: ChallengeRequest,
                      session: Session = Depends(get_session),
                      user: User = Depends(get_current_user)) -> ChallengeResponse:
    device = session.get(Device, body.serial)
    if device is None:
        raise HTTPException(status_code=404, detail=f"Device '{body.serial}' not found")

    if device.last_attested_at is not None:
        elapsed = utcnow() - device.last_attested_at
        cooldown = timedelta(seconds=DEVICE_ATTEST_COOLDOWN_SECONDS)
        if elapsed < cooldown:
            retry = int((cooldown - elapsed).total_seconds())
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Ownership recently changed; retry in {retry}s",
                headers={"Retry-After": str(max(retry, 1))},
            )

    instance_id = str(uuid.uuid4())
    nonce = secrets.token_bytes(NONCE_LENGTH)
    server_time = int(utcnow().timestamp())
    challenge.put(instance_id, {
        "serial": device.serial,
        "nonce": base64.b64encode(nonce).decode(),
        "server_time": server_time,
        "user_id": user.id,
    })
    return ChallengeResponse(
        instance_id=instance_id,
        nonce=base64.b64encode(nonce).decode(),
        server_time=server_time,
        user_id=user.id,
    )


@router.post("/attest")
def attest(body: AttestRequest,
           session: Session = Depends(get_session),
           user: User = Depends(get_current_user)):
    stored = challenge.take(body.instance_id)
    if stored is None:
        raise HTTPException(status_code=410, detail="Challenge not found or expired")
    if stored["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Challenge belongs to another user")

    device = session.get(Device, stored["serial"])
    if device is None:
        raise HTTPException(status_code=404, detail="Device no longer exists")

    payload = challenge.build_payload(
        base64.b64decode(stored["nonce"]),
        body.instance_id,
        stored["server_time"],
        stored["user_id"],
        stored["serial"],
    )
    pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), device.public_key)
    try:
        pub.verify(base64.b64decode(body.signature), payload, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        # Failed attestation does not consume the per-device cooldown.
        raise HTTPException(status_code=400, detail="Signature verification failed")

    device.owner_id = user.id
    device.last_attested_at = utcnow()
    session.add(device)
    session.commit()
    return {"serial": device.serial, "owner_id": user.id}
