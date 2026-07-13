"""Tests for the /model/secure/* routes (api.routes.secure).

Like test_model.py these stop at the enqueue boundary: no worker runs, so the
aggregation task is never exercised here (that path needs TensorFlow and is
covered by the secure path of scripts.fed_client). Sealing a round — normally done by the
harness against the DB — is reproduced directly so the sealed-only endpoints can
be reached without a full cohort."""

import base64

import pytest
from sqlmodel import select

from common.db import (
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    Session,
    engine,
    utcnow,
)
from common.secure_agg import compute_scale, generate_keypair

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


def _join(client, headers, key):
    _, pk = generate_keypair()
    return client.post(f"/model/secure/join/{key}", headers=headers,
                       json={"ka_public_key": base64.b64encode(pk).decode()})


def _seal(round_id: int) -> int:
    with Session(engine) as session:
        rnd = session.get(SecureRound, round_id)
        members = list(session.exec(
            select(SecureRoundMember).where(SecureRoundMember.round_id == round_id)))
        rnd.member_count = len(members)
        rnd.scale = compute_scale(len(members), rnd.clip_bound)
        rnd.status = SecureRoundStatus.sealed
        rnd.sealed_at = utcnow()
        session.add(rnd)
        session.commit()
        return len(members)


def test_join_requires_device_owner(client, auth_headers, deviceless_auth_headers):
    model = _secure_model(client, auth_headers)
    assert _join(client, deviceless_auth_headers, model["key"]).status_code == 403


def test_join_rejects_non_secure_model(client, auth_headers, owned_device):
    model = _nonsecure_model(client, auth_headers)
    assert _join(client, auth_headers, model["key"]).status_code == 404


def test_join_bad_key_400(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    resp = client.post(f"/model/secure/join/{model['key']}", headers=auth_headers,
                       json={"ka_public_key": base64.b64encode(b"\x04short").decode()})
    assert resp.status_code == 400


def test_join_creates_round(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    resp = _join(client, auth_headers, model["key"])
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["round_id"] > 0
    assert body["base_weights_id"] > 0
    assert body["user_id"] > 0


def test_descriptor_before_seal_409(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    round_id = _join(client, auth_headers, model["key"]).json()["round_id"]
    resp = client.get(f"/model/secure/round/{round_id}", headers=auth_headers)
    assert resp.status_code == 409


def test_submit_before_seal_409(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    round_id = _join(client, auth_headers, model["key"]).json()["round_id"]
    resp = client.post(f"/model/secure/submit/{round_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=b"\x00" * (model["weight_count"] * 4))
    assert resp.status_code == 409


def test_descriptor_non_member_404(client, auth_headers, deviceless_auth_headers,
                                   owned_device):
    model = _secure_model(client, auth_headers)
    round_id = _join(client, auth_headers, model["key"]).json()["round_id"]
    _seal(round_id)
    # A user who never joined can't even learn the round exists.
    resp = client.get(f"/model/secure/round/{round_id}", headers=deviceless_auth_headers)
    assert resp.status_code == 404


def test_descriptor_after_seal_carries_roster(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    join = _join(client, auth_headers, model["key"]).json()
    n = _seal(join["round_id"])
    resp = client.get(f"/model/secure/round/{join['round_id']}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    desc = resp.json()
    assert desc["member_count"] == n
    assert desc["weight_count"] == model["weight_count"]
    assert desc["ring_modulus"] == 2 ** 32
    assert desc["scale"] > 0
    member_ids = {e["user_id"] for e in desc["roster"]}
    assert join["user_id"] in member_ids


def test_submit_masked_once(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    round_id = _join(client, auth_headers, model["key"]).json()["round_id"]
    _seal(round_id)
    url = f"/model/secure/submit/{round_id}"
    body = b"\x00" * (model["weight_count"] * 4)

    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=body).status_code == 202
    # A second masked vector under the same masks would leak a difference — rejected.
    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=body).status_code == 409


def test_submit_wrong_length_400(client, auth_headers, owned_device):
    model = _secure_model(client, auth_headers)
    round_id = _join(client, auth_headers, model["key"]).json()["round_id"]
    _seal(round_id)
    resp = client.post(f"/model/secure/submit/{round_id}",
                       headers=auth_headers | OCTET_STREAM, content=b"\x00" * 8)
    assert resp.status_code == 400
