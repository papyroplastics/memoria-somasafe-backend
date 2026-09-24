import uuid
from contextlib import contextmanager

from common.redis import client

_release = client.register_script("""
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
""")


@contextmanager
def model_lock(name: str, ttl: int):
    key = f"lock:{name}"
    token = uuid.uuid4().hex
    held = bool(client.set(key, token, nx=True, ex=ttl))
    try:
        yield held
    finally:
        if held:
            _release(keys=[key], args=[token])
