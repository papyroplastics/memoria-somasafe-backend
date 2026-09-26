from fastapi import HTTPException, status

from common.ratelimit import RateLimit, over_limit


def check_limit(action: RateLimit, user_id: int, resource: str, limit: int,
                window: int) -> None:
    """429 if already at ``limit``; spend the slot afterwards with ``add_usage``."""
    ttl = over_limit(action, user_id, resource, limit, window)
    if ttl is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limited; retry in {ttl}s",
            headers={"Retry-After": str(max(ttl, 1))},
        )
