"""Shared Redis connection (rate-limit db, separate from the Celery broker).

Both the rate limiter and the device-attestation challenge store key into this
one client; see ``REDIS_URL`` in ``common.config``.
"""

import redis

from common.config import REDIS_URL

client = redis.from_url(REDIS_URL)
