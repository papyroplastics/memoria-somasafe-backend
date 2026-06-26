"""Shared fixtures for the API tests.

These run against the **configured** database and Redis (the real services), and
assume the seed script has already been run (``make seed``) so the model
registry and the ``SEED_USER`` account exist. Tests add their own throwaway
devices (random serial + key) and clear the rate-limit counters between runs, so
nothing here imports TensorFlow or the ml package — the suite stays fast.

A broker is set to ``memory://`` before the app is imported: the model routes
enqueue quantization tasks but a worker is never run, so submissions stay
``pending`` and the tests assert up to the enqueue/poll boundary.
"""

import os

os.environ.setdefault("BROKER_URL", "memory://")

import secrets

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from sqlmodel import select

from api.lib import ratelimit
from api.main import app
from common.config import SEED_PASSWORD, SEED_USER
from common.db import Device, Session, User, engine, utcnow


def pub_point(priv: ec.EllipticCurvePrivateKey) -> bytes:
    """The device's 65-byte uncompressed P-256 public point, as stored on Device."""
    return priv.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )


@pytest.fixture(scope="session")
def client():
    # The context manager runs the lifespan (init_db) before the first request.
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Start every test with a clean rate-limit slate so cooldown/quota assertions
    are independent of prior tests and prior runs."""
    ratelimit.reset()
    yield


@pytest.fixture
def seed_user_id() -> int:
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == SEED_USER)).first()
        assert user is not None, "seed the database first (make seed)"
        return user.id


@pytest.fixture
def auth_headers(client) -> dict:
    resp = client.post("/auth/token",
                       data={"username": SEED_USER, "password": SEED_PASSWORD})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def owned_device(seed_user_id):
    """A throwaway device already attested to the seed user — unlocks the
    device-owner-gated model routes. Removed on teardown."""
    serial = f"SN-TEST-{secrets.token_hex(4)}"
    priv = ec.generate_private_key(ec.SECP256R1())
    with Session(engine) as session:
        session.add(Device(serial=serial, public_key=pub_point(priv),
                           owner_id=seed_user_id, last_attested_at=utcnow()))
        session.commit()
    yield serial
    _delete_device(serial)


@pytest.fixture
def unclaimed_device():
    """A throwaway ownerless device whose private key the test holds, so it can
    sign a real challenge and exercise the attestation flow. Removed on teardown."""
    serial = f"SN-ATTEST-{secrets.token_hex(4)}"
    priv = ec.generate_private_key(ec.SECP256R1())
    with Session(engine) as session:
        session.add(Device(serial=serial, public_key=pub_point(priv)))
        session.commit()
    yield serial, priv
    _delete_device(serial)


def _delete_device(serial: str) -> None:
    with Session(engine) as session:
        device = session.get(Device, serial)
        if device is not None:
            session.delete(device)
            session.commit()
