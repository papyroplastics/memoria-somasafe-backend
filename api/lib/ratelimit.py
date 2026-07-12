"""HTTP rate limiting for the gateway: thin wrappers over the two-phase Redis
primitives in ``common.ratelimit``. A route checks the limit up front
(``check_limit`` -> 429 with ``Retry-After``), does its work, and spends a slot
(``record_usage``) only afterwards, so a rejected or no-op request is not counted.
The keying and Redis mechanics live in ``common`` so the worker can reuse them
without importing API code.
"""

from fastapi import HTTPException, status

from common.ratelimit import RateLimit, add_usage, over_limit


def check_limit(action: RateLimit, user_id: int, resource: str, limit: int,
                window: int) -> None:
    """Reject with 429 if this (user, resource) is already at ``limit`` for the
    window. Does not spend a slot — call ``record_usage`` after the work succeeds
    (or fails in a way that should still count). ``limit=1`` is a cooldown."""
    ttl = over_limit(action, user_id, resource, limit, window)
    if ttl is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limited; retry in {ttl}s",
            headers={"Retry-After": str(max(ttl, 1))},
        )


def record_usage(action: RateLimit, user_id: int, resource: str,
                 window: int) -> None:
    """Spend one slot for this (user, resource) — the second phase of the limit,
    run after the request has done its work."""
    add_usage(action, user_id, resource, window)
