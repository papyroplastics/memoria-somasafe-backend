"""Device attestation challenges: the canonical signed payload plus a one-shot
Redis store for the in-flight challenges.

A challenge is issued by ``POST /device/challenge`` and consumed exactly once by
``POST /device/attest``. Entries share the rate-limit Redis db and carry a TTL,
so an unanswered challenge simply expires.

The canonical payload the device signs is a fixed concatenation (big-endian, no
separators): nonce(32B) ‖ instance_id(16B uuid) ‖ server_time(u64) ‖
user_id(u64) ‖ serial(ascii).
"""

import json
import uuid
from sqlmodel import Session
from fastapi import HTTPException

from common.config import DEVICE_CHALLENGE_TTL_SECONDS
from common.redis import client
from common.db import user_owns_device, User

_PREFIX = "device:challenge:"


def build_payload(nonce: bytes, instance_id: str, server_time: int,
                  user_id: int, serial: str) -> bytes:
    return (
        nonce
        + uuid.UUID(instance_id).bytes
        + server_time.to_bytes(8, "big")
        + user_id.to_bytes(8, "big")
        + serial.encode("ascii")
    )


def put(instance_id: str, data: dict) -> None:
    client.set(
        _PREFIX + instance_id, json.dumps(data), ex=DEVICE_CHALLENGE_TTL_SECONDS)


def take(instance_id: str) -> dict | None:
    """Atomically fetch and delete a challenge (one-shot). None if absent/expired."""
    raw = client.getdel(_PREFIX + instance_id)
    return json.loads(raw) if raw is not None else None

def require_device_owner(session: Session, user: User) -> None:
    if not user_owns_device(session, user.id):
        raise HTTPException(status_code=403, detail="No attested device for this user")



