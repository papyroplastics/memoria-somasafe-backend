import redis
from sqlalchemy import text

from common.config import BROKER_URL, REDIS_URL
from common.db import engine

TRUNCATED = ("quantizationresult", "quantizationjob", "clientdeltasubmission",
             "securepartial", "securesessionmember", "securesession", "authsession")
KOMBU_BINDINGS = b"_kombu.binding."


def reset_database() -> int:
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TRUNCATED)} RESTART IDENTITY"))
        deleted = conn.execute(text(
            "DELETE FROM globalweights WHERE id NOT IN "
            "(SELECT min(id) FROM globalweights GROUP BY version_id)")).rowcount
        conn.execute(text(
            "SELECT setval(pg_get_serial_sequence('globalweights', 'id'), "
            "(SELECT coalesce(max(id), 0) + 1 FROM globalweights), false)"))
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("VACUUM ANALYZE globalweights, weightsartifact"))
    return deleted


def reset_broker(client: redis.Redis) -> int:
    keys = [key for key in client.scan_iter(count=1000) if not key.startswith(KOMBU_BINDINGS)]
    for start in range(0, len(keys), 1000):
        client.delete(*keys[start:start + 1000])
    return len(keys)


def main() -> None:
    weights = reset_database()
    print(f"truncated {', '.join(TRUNCATED)}; dropped {weights} aggregated weight snapshot(s)")

    store = redis.from_url(REDIS_URL)
    store_keys = store.dbsize()
    store.flushdb()
    print(f"flushed {store_keys} key(s) from {REDIS_URL}")

    print(f"dropped {reset_broker(redis.from_url(BROKER_URL))} key(s) from {BROKER_URL}")


if __name__ == "__main__":
    main()
