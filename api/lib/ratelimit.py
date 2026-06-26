"""Redis-backed per-user, per-model rate limiting for the gateway.

Two primitives, both atomic on the Redis side:
- cooldown: at most one success per window (SET NX EX).
- daily quota: a capped counter over a rolling window (INCR + EXPIRE).

A breach raises HTTP 429 with a Retry-After header.
"""

import redis
from fastapi import HTTPException, status

from common.config import RATELIMIT_URL

_client = redis.from_url(RATELIMIT_URL)


def _too_many(retry_after: int, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(retry_after, 1))},
    )


def _key(action: str, user_id: int, model_key: str) -> str:
    return f"rl:{action}:{user_id}:{model_key}"


def enforce_cooldown(action: str, user_id: int, model_key: str,
                     window: int) -> None:
    """Allow one request per ``window`` seconds for this (user, model). A repeat
    within the window fails until the key expires."""
    key = _key(action, user_id, model_key)
    if not _client.set(key, 1, nx=True, ex=window):
        ttl = _client.ttl(key)
        raise _too_many(ttl, f"Rate limited; retry in {ttl}s")


def enforce_daily_quota(action: str, user_id: int, model_key: str,
                        limit: int, window: int) -> None:
    """Allow up to ``limit`` requests per ``window`` seconds for this (user,
    model). The window starts on the first request and expires as a whole."""
    key = _key(action, user_id, model_key)
    count = _client.incr(key)
    if count == 1:
        _client.expire(key, window)
    if count > limit:
        ttl = _client.ttl(key)
        raise _too_many(ttl, f"Daily limit of {limit} reached; retry in {ttl}s")


def reset() -> None:
    """Drop every ``rl:`` counter. Test helper: lets the rate-limited endpoints
    be exercised repeatedly against a shared Redis without waiting out windows."""
    keys = list(_client.scan_iter(match="rl:*"))
    if keys:
        _client.delete(*keys)
