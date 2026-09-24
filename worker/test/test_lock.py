import uuid

from common.redis import client
from worker.locking import model_lock


def test_lock_is_exclusive():
    name = f"test:{uuid.uuid4().hex}"
    with model_lock(name, 10) as held:
        assert held
        with model_lock(name, 10) as second:
            assert not second
    with model_lock(name, 10) as again:
        assert again


def test_release_keeps_successor_lock():
    name = f"test:{uuid.uuid4().hex}"
    key = f"lock:{name}"
    try:
        with model_lock(name, 10) as held:
            assert held
            client.set(key, "successor")
        assert client.get(key) == b"successor"
    finally:
        client.delete(key)
