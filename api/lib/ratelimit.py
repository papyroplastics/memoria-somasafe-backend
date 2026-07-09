"""HTTP rate limiting for the gateway: thin wrappers that turn a Redis-backed
breach (``common.ratelimit``) into a 429 with a ``Retry-After`` header. The keying
and Redis mechanics live in ``common`` so the worker can reuse them without
importing API code.
"""

from fastapi import HTTPException, status

from common.ratelimit import hit_quota, try_cooldown


def _too_many(retry_after: int, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(retry_after, 1))},
    )


def enforce_cooldown(action: str, user_id: int, model_key: str,
                     window: int) -> None:
    """Allow one request per ``window`` seconds for this (user, model). A repeat
    within the window fails until the key expires."""
    ttl = try_cooldown(action, user_id, model_key, window)
    if ttl is not None:
        raise _too_many(ttl, f"Rate limited; retry in {ttl}s")


def enforce_daily_quota(action: str, user_id: int, model_key: str,
                        limit: int, window: int) -> None:
    """Allow up to ``limit`` requests per ``window`` seconds for this (user,
    model). The window starts on the first request and expires as a whole."""
    ttl = hit_quota(action, user_id, model_key, limit, window)
    if ttl is not None:
        raise _too_many(ttl, f"Daily limit of {limit} reached; retry in {ttl}s")
