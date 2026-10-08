from enum import Enum

from common.redis import client


class RateLimit(str, Enum):
    """A rate-limited action. The value is the Redis key segment, so it is stable
    across the api (which enforces the limit) and the worker (which clears it)."""

    model_download = "download"
    weights_download = "weights"
    weight_submit = "submit"
    ota_download = "ota-download"
    secure_join = "secure-join"


# Model-scoped actions, keyed by model key — the ones a federated round clears so
# clients can immediately re-pull and re-submit. (``ota_download`` is per-interface
# and unrelated to a model round.)
_MODEL_ACTIONS = (RateLimit.model_download, RateLimit.weights_download,
                  RateLimit.weight_submit, RateLimit.secure_join)


def _key(action: RateLimit, user_id: int, resource: str) -> str:
    return f"rl:{action.value}:{resource}:{user_id}"


def over_limit(action: RateLimit, user_id: int, resource: str, limit: int,
               window: int) -> int | None:
    """Peek without spending: ``None`` while the (user, resource) counter is below
    ``limit``, otherwise the remaining TTL (seconds) until the window clears."""
    key = _key(action, user_id, resource)
    count = int(client.get(key) or 0)
    if count < limit:
        return None
    ttl = client.ttl(key)
    return ttl if ttl and ttl > 0 else window


def add_usage(action: RateLimit, user_id: int, resource: str, window: int) -> None:
    """Spend one slot for this (user, resource). The window starts on the first
    hit and expires as a whole."""
    key = _key(action, user_id, resource)
    if client.incr(key) == 1:
        client.expire(key, window)


def clear_model_limits(model_key: str) -> None:
    for action in _MODEL_ACTIONS:
        keys = list(client.scan_iter(match=f"rl:{action.value}:{model_key}:*"))
        if keys:
            client.delete(*keys)


def clear_user_limits(model_key: str, user_ids: list[int]) -> None:
    keys = [_key(action, user_id, model_key) for user_id in user_ids
            for action in (RateLimit.secure_join, RateLimit.weight_submit)]
    if keys:
        client.delete(*keys)


def reset() -> None:
    """Drop every ``rl:`` counter. Test helper: lets the rate-limited endpoints
    be exercised repeatedly against a shared Redis without waiting out windows."""
    keys = list(client.scan_iter(match="rl:*"))
    if keys:
        client.delete(*keys)
