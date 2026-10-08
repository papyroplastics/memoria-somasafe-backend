"""Tests for the /model/secure/* routes (api.routes.secure). No worker runs, so
sessions are sealed directly against the DB to reach the sealed-only endpoints."""

import base64
import threading

import pytest
from sqlalchemy import delete, func, update
from sqlmodel import select

from common import ratelimit
from common.db import (
    GlobalWeights,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    Session,
    engine,
    get_latest_weights,
    utcnow,
)
from common.secure_agg import generate_keypair
from common.secure_session import seal_session
from ..routes import secure as secure_routes
from ..routes.secure import _open_session

OCTET_STREAM = {"Content-Type": "application/octet-stream"}


def _secure_model(client, headers) -> dict:
    resp = client.get("/model/list", headers=headers)
    assert resp.status_code == 200, resp.text
    for model in resp.json():
        if model["weights_version"] is not None and model["submission_type"] == "secure":
            return model
    pytest.skip("no seeded secure model has weights; run the seed script first")


def _nonsecure_model(client, headers) -> dict:
    resp = client.get("/model/list", headers=headers)
    for model in resp.json():
        if model["weights_version"] is not None and model["submission_type"] != "secure":
            return model
    pytest.skip("no seeded non-secure model has weights")


def _active_weights_id(key: str) -> int:
    with Session(engine) as session:
        return get_latest_weights(session, key).id


def _user_headers(client, i: int) -> dict:
    resp = client.post("/auth/token", data={"username": f"test_{i}", "password": f"test_{i}"})
    if resp.status_code != 200:
        pytest.skip("seed the test users first (make db-seed)")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _join(client, headers, key, weights_id=None):
    weights_id = _active_weights_id(key) if weights_id is None else weights_id
    _, pk = generate_keypair()
    return client.post(f"/model/secure/join/{key}/{weights_id}", headers=headers,
                       json={"ka_public_key": base64.b64encode(pk).decode()})


def _seal(session_id: int) -> int:
    with Session(engine) as session:
        n = seal_session(session, session_id)
        session.commit()
        return n


def _status(session_id: int) -> SecureSessionStatus:
    with Session(engine) as session:
        return session.get(SecureSession, session_id).status


@pytest.fixture(autouse=True)
def _isolate_sessions():
    live = (SecureSessionStatus.open, SecureSessionStatus.sealed)
    with Session(engine) as session:
        stale = select(SecureSession.id).where(SecureSession.status.in_(live))  # type: ignore
        session.execute(delete(SecureSessionMember)
                        .where(SecureSessionMember.session_id.in_(stale)))  # type: ignore
        session.execute(update(SecureSession)
                        .where(SecureSession.status.in_(live))  # type: ignore
                        .values(status=SecureSessionStatus.failed, finished_at=utcnow()))
        floor = session.exec(select(func.max(SecureSession.id))).one() or 0
        session.commit()
    yield
    with Session(engine) as session:
        session.execute(delete(SecureSessionMember)
                        .where(SecureSessionMember.session_id > floor))  # type: ignore
        session.execute(delete(SecureSession).where(SecureSession.id > floor))  # type: ignore
        session.commit()


def test_join_requires_device_owner(client, auth_headers, deviceless_auth_headers):
    model = _secure_model(client, auth_headers)
    assert _join(client, deviceless_auth_headers, model["key"]).status_code == 403


def test_join_rejects_non_secure_model(client, auth_headers, owned_device):
    model = _nonsecure_model(client, auth_headers)
    assert _join(client, auth_headers, model["key"]).status_code == 404


def test_join_bad_key_400(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    weights_id = _active_weights_id(model["key"])
    resp = client.post(f"/model/secure/join/{model['key']}/{weights_id}", headers=auth_headers,
                       json={"ka_public_key": base64.b64encode(b"\x04short").decode()})
    assert resp.status_code == 400


def test_join_unknown_weights_400(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    assert _join(client, auth_headers, model["key"], 999999999).status_code == 400


def test_join_stale_weights_409(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    with Session(engine) as session:
        active = get_latest_weights(session, model["key"])
        stale = GlobalWeights(model_key=active.model_key, version_id=active.version_id,
                              weights=b"w", valid=False)
        session.add(stale)
        session.commit()
        stale_id = stale.id
    try:
        assert _join(client, auth_headers, model["key"], stale_id).status_code == 409
    finally:
        with Session(engine) as session:
            session.delete(session.get(GlobalWeights, stale_id))
            session.commit()


def test_join_creates_session(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    resp = _join(client, auth_headers, model["key"])
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["session_id"] > 0
    assert body["base_weights_id"] == _active_weights_id(model["key"])
    assert _status(body["session_id"]) is SecureSessionStatus.open


def test_rejoin_while_open_keeps_seat(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    first = _join(client, auth_headers, model["key"]).json()
    ratelimit.reset()
    second = _join(client, auth_headers, model["key"])
    assert second.status_code == 202, second.text
    assert second.json()["session_id"] == first["session_id"]


def test_join_seals_at_max_members(client, auth_headers, monkeypatch):
    monkeypatch.setattr(secure_routes, "SECURE_SESSION_MAX_MEMBERS", 2)
    key = _secure_model(client, auth_headers)["key"]
    first = _join(client, _user_headers(client, 1), key).json()
    second = _join(client, _user_headers(client, 2), key).json()
    assert first["session_id"] == second["session_id"]
    assert _status(first["session_id"]) is SecureSessionStatus.sealed

    third = _join(client, _user_headers(client, 3), key)
    assert third.status_code == 202, third.text
    assert third.json()["session_id"] != first["session_id"]


def test_seat_in_sealed_session_409(client, auth_headers, owned_device, monkeypatch):
    monkeypatch.setattr(secure_routes, "SECURE_SESSION_MAX_MEMBERS", 1)
    model = _secure_model(client, auth_headers)
    session_id = _join(client, auth_headers, model["key"]).json()["session_id"]
    assert _status(session_id) is SecureSessionStatus.sealed
    ratelimit.reset()
    assert _join(client, auth_headers, model["key"]).status_code == 409


def test_concurrent_first_joins_share_one_session(client, auth_headers):
    key = _secure_model(client, auth_headers)["key"]
    with Session(engine) as session:
        weights = get_latest_weights(session, key)

    barrier = threading.Barrier(8)
    session_ids = []

    def open_one():
        with Session(engine) as session:
            barrier.wait()
            session_ids.append(_open_session(session, weights).id)
            session.commit()

    threads = [threading.Thread(target=open_one) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(session_ids) == 8 and len(set(session_ids)) == 1
    with Session(engine) as session:
        open_sessions = session.exec(select(SecureSession).where(
            SecureSession.base_weights_id == weights.id,
            SecureSession.status == SecureSessionStatus.open)).all()
    assert [s.id for s in open_sessions] == session_ids[:1]


def test_concurrent_joins_fill_sealed_sessions(client, auth_headers, monkeypatch):
    monkeypatch.setattr(secure_routes, "SECURE_SESSION_MAX_MEMBERS", 3)
    key = _secure_model(client, auth_headers)["key"]
    headers = [_user_headers(client, i) for i in range(1, 7)]
    barrier = threading.Barrier(len(headers))
    responses = []

    def join(h):
        barrier.wait()
        responses.append(_join(client, h, key))

    threads = [threading.Thread(target=join, args=(h,)) for h in headers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [r.status_code for r in responses] == [202] * len(headers)
    session_ids = {r.json()["session_id"] for r in responses}
    assert len(session_ids) == 2
    with Session(engine) as session:
        for session_id in session_ids:
            row = session.get(SecureSession, session_id)
            assert row.status is SecureSessionStatus.sealed and row.member_count == 3


def test_descriptor_before_seal_409(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    session_id = _join(client, auth_headers, model["key"]).json()["session_id"]
    resp = client.get(f"/model/secure/session/{session_id}", headers=auth_headers)
    assert resp.status_code == 409


def test_submit_before_seal_409(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    session_id = _join(client, auth_headers, model["key"]).json()["session_id"]
    resp = client.post(f"/model/secure/submit/{session_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=b"\x00" * (model["weight_count"] * 4))
    assert resp.status_code == 409


def test_descriptor_non_member_404(client, auth_headers, deviceless_auth_headers,
                                   owned_device):
    model = _secure_model(client, auth_headers)
    session_id = _join(client, auth_headers, model["key"]).json()["session_id"]
    _seal(session_id)
    resp = client.get(f"/model/secure/session/{session_id}", headers=deviceless_auth_headers)
    assert resp.status_code == 404


def test_descriptor_after_seal_carries_roster(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    join = _join(client, auth_headers, model["key"]).json()
    n = _seal(join["session_id"])
    resp = client.get(f"/model/secure/session/{join['session_id']}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    desc = resp.json()
    assert desc["session_id"] == join["session_id"]
    assert desc["member_count"] == n
    assert desc["weight_count"] == model["weight_count"]
    assert desc["ring_modulus"] == 2 ** 32
    assert desc["scale"] > 0
    assert join["user_id"] in {e["user_id"] for e in desc["roster"]}


def test_submit_masked_once(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    session_id = _join(client, auth_headers, model["key"]).json()["session_id"]
    _seal(session_id)
    url = f"/model/secure/submit/{session_id}"
    body = b"\x00" * (model["weight_count"] * 4)

    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=body).status_code == 202
    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=body).status_code == 409


def test_submit_wrong_length_400(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    session_id = _join(client, auth_headers, model["key"]).json()["session_id"]
    _seal(session_id)
    resp = client.post(f"/model/secure/submit/{session_id}",
                       headers=auth_headers | OCTET_STREAM, content=b"\x00" * 8)
    assert resp.status_code == 400
