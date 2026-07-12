"""Redis-backed rate-limit primitives, shared so the worker never imports API code.

The limit is a capped counter over a rolling window, keyed per (action, user,
resource), split into two phases so a request only spends quota once it has done
the work it was admitted for:
- ``over_limit``: peek — is the counter already at ``limit``? (a read, no mutation)
- ``add_usage``: spend one slot (INCR + EXPIRE on the first hit of the window)

The gateway checks ``over_limit`` up front (turning a breach into an HTTP 429, see
``api.lib.ratelimit``), runs the request, and calls ``add_usage`` only afterwards,
so a rejected or no-op request is not counted. Splitting the check from the spend
makes the limit soft under concurrency (two simultaneous requests can both pass the
peek), which is fine at this scale. A cooldown is just this with ``limit=1``.

``clear_model_limits`` is used by the worker after a federated round; ``reset`` is a
test helper.
"""

from enum import Enum

from common.redis import client


class RateLimit(str, Enum):
    """A rate-limited action. The value is the Redis key segment, so it is stable
    across the api (which enforces the limit) and the worker (which clears it)."""

    model_download = "download"
    weight_submit = "submit"
    ota_download = "ota-download"


# Model-scoped actions, keyed by model key — the ones a federated round clears so
# clients can immediately re-pull and re-submit. (``ota_download`` is per-interface
# and unrelated to a model round.)
_MODEL_ACTIONS = (RateLimit.model_download, RateLimit.weight_submit)


def _key(action: RateLimit, user_id: int, resource: str) -> str:
    return f"rl:{action.value}:{user_id}:{resource}"


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
    """Drop the download/quantize/submit counters for a model across all users.
    Called after a federated round produces new weights so every client can
    immediately re-pull the updated model and submit again."""
    for action in _MODEL_ACTIONS:
        keys = list(client.scan_iter(match=f"rl:{action.value}:*:{model_key}"))
        if keys:
            client.delete(*keys)


def reset() -> None:
    """Drop every ``rl:`` counter. Test helper: lets the rate-limited endpoints
    be exercised repeatedly against a shared Redis without waiting out windows."""
    keys = list(client.scan_iter(match="rl:*"))
    if keys:
        client.delete(*keys)
