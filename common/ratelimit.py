"""Redis-backed rate-limit primitives, shared so the worker never imports API code.

Two atomic operations keyed per (action, user, model):
- ``try_cooldown``: at most one success per window (SET NX EX).
- ``hit_quota``: a capped counter over a rolling window (INCR + EXPIRE).

Both return the remaining TTL on a breach (``None`` otherwise); the gateway turns
that into an HTTP 429 (see ``api.lib.ratelimit``). ``clear_model_limits`` is used by
the worker after a federated round; ``reset`` is a test helper.
"""

from common.redis import client


def _key(action: str, user_id: int, model_key: str) -> str:
    return f"rl:{action}:{user_id}:{model_key}"


def try_cooldown(action: str, user_id: int, model_key: str,
                 window: int) -> int | None:
    """Claim the one-per-``window`` slot for this (user, model). Returns ``None``
    on success, or the remaining TTL (seconds) if the window is still active."""
    key = _key(action, user_id, model_key)
    if client.set(key, 1, nx=True, ex=window):
        return None
    return client.ttl(key)


def hit_quota(action: str, user_id: int, model_key: str, limit: int,
              window: int) -> int | None:
    """Count one request against the rolling ``window`` for this (user, model).
    Returns ``None`` while within ``limit``, or the remaining TTL once the cap is
    exceeded. The window starts on the first request and expires as a whole."""
    key = _key(action, user_id, model_key)
    count = client.incr(key)
    if count == 1:
        client.expire(key, window)
    if count > limit:
        return client.ttl(key)
    return None


def clear_model_limits(model_key: str) -> None:
    """Drop the artifact-download and submission counters for a model across all
    users. Called after a federated round produces new weights so every client
    can immediately re-pull the updated model and submit again."""
    for action in ("download", "quantize", "submit"):
        keys = list(client.scan_iter(match=f"rl:{action}:*:{model_key}"))
        if keys:
            client.delete(*keys)


def reset() -> None:
    """Drop every ``rl:`` counter. Test helper: lets the rate-limited endpoints
    be exercised repeatedly against a shared Redis without waiting out windows."""
    keys = list(client.scan_iter(match="rl:*"))
    if keys:
        client.delete(*keys)
