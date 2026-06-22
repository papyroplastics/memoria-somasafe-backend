import os
import tempfile

# Configure a throwaway SQLite DB and an eager (broker-less) Celery before the
# app modules read these at import time.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{tempfile.gettempdir()}/somasafe_test.db")
os.environ.setdefault("BROKER_URL", "memory://")

import base64

import fakeredis
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from sqlmodel import select
from ai_edge_litert.compiled_model import CompiledModel

from worker.celery_app import app as celery_app

celery_app.conf.task_always_eager = True
import worker.tasks  # noqa: F401,E402 - registers quantize_submission for eager run

import common.ratelimit
from common.db import Device, Session, User, engine, init_db
from api.auth import hash_password
from api.device import _payload
from api.main import app
from scripts.db_seed import seed_models

# Rate limiting talks to Redis; swap in an in-memory fake for the test.
common.ratelimit._client = fakeredis.FakeStrictRedis()

init_db()

TEST_USER = "tester"
TEST_PASSWORD = "testpass"


def _pub_point(priv: ec.EllipticCurvePrivateKey) -> bytes:
    return priv.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )


with Session(engine) as _session:
    if _session.get(User, 1) is None:
        _session.add(User(username=TEST_USER, hashed_password=hash_password(TEST_PASSWORD)))
        _session.commit()
    # The model endpoints require an attested device; give the default user one.
    if _session.get(Device, "SN000000OWNED") is None:
        _session.add(Device(serial="SN000000OWNED",
                            public_key=_pub_point(ec.generate_private_key(ec.SECP256R1())),
                            owner_id=1))
        _session.commit()
    # Seed the model registry (fingerprint + metadata + initial weights) from the
    # trained artifacts, the same path scripts.db_seed uses.
    seed_models(_session)

client = TestClient(app)


def _auth_headers() -> dict:
    resp = client.post("/auth/token",
                       data={"username": TEST_USER, "password": TEST_PASSWORD})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_auth_required():
    assert client.get("/model/list").status_code == 401


def test_quantize_feature_mlp():
    headers = _auth_headers()

    list_response = client.get("/model/list", headers=headers)
    assert list_response.status_code == 200
    model_info = next(m for m in list_response.json() if m["key"] == "feature-mlp")
    fingerprint = model_info["fingerprint"]

    response = client.get("/model/download/trainable/feature-mlp", headers=headers)
    assert response.status_code == 200
    assert response.headers["X-Model-Fingerprint"] == fingerprint

    # The base weights snapshot the client trains from (id echoed back to quantize).
    weights_resp = client.get("/model/weights/feature-mlp", headers=headers)
    assert weights_resp.status_code == 200
    assert weights_resp.headers["X-Model-Fingerprint"] == fingerprint
    weights_id = int(weights_resp.headers["X-Weights-ID"])

    compiled = CompiledModel.from_buffer(response.content)
    out_buf = compiled.create_output_buffer_by_name('save', 'parameters')
    num_elements = int(np.prod(out_buf.get_tensor_details()['shape']))
    compiled.run_by_name('save', {}, {'parameters': out_buf})
    parameters = out_buf.read(num_elements, np.float32)

    # Submit weights -> 202 + job id; with eager Celery the worker runs inline.
    submit = client.post(
        "/model/quantize/feature-mlp",
        json={'parameters': parameters.tolist(), 'weights_id': weights_id},
        headers=headers,
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    # Poll the result endpoint for the quantized .tflite.
    result = client.get(f"/model/quantize/result/{job_id}", headers=headers)
    assert result.status_code == 200
    assert len(result.content) > 0


def _make_user(username: str) -> None:
    with Session(engine) as session:
        if session.exec(select(User).where(User.username == username)).first() is None:
            session.add(User(username=username, hashed_password=hash_password(username)))
            session.commit()


def _headers_for(username: str) -> dict:
    resp = client.post("/auth/token", data={"username": username, "password": username})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _add_device(serial: str, priv: ec.EllipticCurvePrivateKey) -> None:
    with Session(engine) as session:
        session.add(Device(serial=serial, public_key=_pub_point(priv)))
        session.commit()


def _sign_challenge(priv: ec.EllipticCurvePrivateKey, ch: dict, serial: str) -> str:
    payload = _payload(base64.b64decode(ch["nonce"]), ch["instance_id"],
                       ch["server_time"], ch["user_id"], serial)
    return base64.b64encode(priv.sign(payload, ec.ECDSA(hashes.SHA256()))).decode()


def test_attestation_unlocks_models():
    serial = "SN0000ATTEST1"
    priv = ec.generate_private_key(ec.SECP256R1())
    _add_device(serial, priv)
    _make_user("attester")
    headers = _headers_for("attester")

    # Gated before attestation.
    assert client.get("/model/download/trainable/feature-mlp",
                      headers=headers).status_code == 403

    ch = client.post("/device/challenge", json={"serial": serial}, headers=headers)
    assert ch.status_code == 200, ch.text
    attest = client.post("/device/attest", headers=headers, json={
        "instance_id": ch.json()["instance_id"],
        "signature": _sign_challenge(priv, ch.json(), serial),
    })
    assert attest.status_code == 200, attest.text

    # Unlocked after attestation.
    assert client.get("/model/download/trainable/feature-mlp",
                      headers=headers).status_code == 200

    # Ownership can only change once a day -> a second challenge is rate limited.
    assert client.post("/device/challenge", json={"serial": serial},
                       headers=headers).status_code == 429


def test_attestation_bad_signature():
    serial = "SN0000BADSIG1"
    priv = ec.generate_private_key(ec.SECP256R1())
    _add_device(serial, priv)
    _make_user("badsigner")
    headers = _headers_for("badsigner")

    ch = client.post("/device/challenge", json={"serial": serial}, headers=headers)
    assert ch.status_code == 200, ch.text
    bad = client.post("/device/attest", headers=headers, json={
        "instance_id": ch.json()["instance_id"],
        "signature": base64.b64encode(b"\x00" * 70).decode(),
    })
    assert bad.status_code == 400

    # A failed attempt neither sets an owner nor consumes the daily window.
    with Session(engine) as session:
        device = session.get(Device, serial)
        assert device.owner_id is None
        assert device.last_attested_at is None
    assert client.post("/device/challenge", json={"serial": serial},
                       headers=headers).status_code == 200


def test_challenge_unknown_device():
    headers = _auth_headers()
    assert client.post("/device/challenge", json={"serial": "SN0000NOSUCH0"},
                       headers=headers).status_code == 404
