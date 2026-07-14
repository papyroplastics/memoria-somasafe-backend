"""Access-token session store: Redis-backed, so validating a Bearer token on
every authed request is a single GET instead of a Postgres round trip (see
common.db.AuthSession for the DB-backed refresh-token side, which stays in
Postgres since it's long-lived and needs no per-request lookup). Shares the
rate-limit Redis db (common/redis.py).

Keyed two ways: ``auth:access:{hash}`` for O(1) validation, and a per-user
``auth:access:idx:{user_id}`` set so logout-all can revoke every live access
token for a user (mirrors api.lib.ratelimit's per-model scan_iter approach,
just via an explicit index since the primary key isn't user-prefixed). Note
rotating a refresh token (api.routes.auth.refresh) does not revoke the
still-live old access token early — it simply rides out its own short TTL.
"""

import hashlib
import json
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session

from common.config import ACCESS_TOKEN_TTL_SECONDS
from common.db import User, get_session
from common.redis import client

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

_ACCESS_PREFIX = "auth:access:"
_INDEX_PREFIX = "auth:access:idx:"


def token_hash(token: str) -> str:
    # Tokens are high-entropy random strings, so a fast digest is sufficient
    # (argon2 is only needed for low-entropy passwords).
    return hashlib.sha256(token.encode()).hexdigest()


def put_access(token: str, user_id: int, session_id: uuid.UUID) -> None:
    h = token_hash(token)
    payload = json.dumps({"user_id": user_id, "session_id": str(session_id)})
    client.set(_ACCESS_PREFIX + h, payload, ex=ACCESS_TOKEN_TTL_SECONDS)
    client.sadd(_INDEX_PREFIX + str(user_id), h)


def lookup_access(token: str) -> dict | None:
    raw = client.get(_ACCESS_PREFIX + token_hash(token))
    return json.loads(raw) if raw is not None else None


def revoke_access(token: str) -> None:
    client.delete(_ACCESS_PREFIX + token_hash(token))


def revoke_all_access(user_id: int) -> None:
    idx_key = _INDEX_PREFIX + str(user_id)
    hashes = client.smembers(idx_key)
    if hashes:
        client.delete(*(_ACCESS_PREFIX + h.decode() for h in hashes))
    client.delete(idx_key)


async def get_current_user(token: str = Depends(oauth2_scheme),
                           session: Session = Depends(get_session)) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    data = lookup_access(token)
    if data is None:
        raise unauthorized

    user = session.get(User, data["user_id"])
    if user is None or user.disabled:
        raise unauthorized
    return user
