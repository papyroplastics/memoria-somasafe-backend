"""Redis-backed store for in-flight device attestation challenges.

A challenge is issued by ``POST /device/challenge`` and consumed exactly once by
``POST /device/attest``. Entries live on the rate-limit Redis db with a TTL, so
an unanswered challenge simply expires.
"""

import json

import common.ratelimit
from common.config import DEVICE_CHALLENGE_TTL_SECONDS

_PREFIX = "device:challenge:"


def put(instance_id: str, data: dict) -> None:
    # Resolve the client lazily so tests can swap in a fake (see common.ratelimit).
    common.ratelimit._client.set(
        _PREFIX + instance_id, json.dumps(data), ex=DEVICE_CHALLENGE_TTL_SECONDS)


def take(instance_id: str) -> dict | None:
    """Atomically fetch and delete a challenge (one-shot). None if absent/expired."""
    raw = common.ratelimit._client.getdel(_PREFIX + instance_id)
    return json.loads(raw) if raw is not None else None
