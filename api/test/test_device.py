"""Tests for the /device attestation routes (api.routes.device)."""

import base64

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from ..lib import challenge
from common.db import Device, Session, engine


def _sign(priv: ec.EllipticCurvePrivateKey, ch: dict, serial: str) -> str:
    payload = challenge.build_payload(
        base64.b64decode(ch["nonce"]), ch["instance_id"],
        ch["server_time"], ch["user_id"], serial)
    return base64.b64encode(priv.sign(payload, ec.ECDSA(hashes.SHA256()))).decode()


def test_challenge_unknown_device_404(client, auth_headers):
    resp = client.post("/device/challenge",
                       json={"serial": "SN-NO-SUCH-DEV"}, headers=auth_headers)
    assert resp.status_code == 404


def test_attestation_sets_owner_and_locks_cooldown(client, auth_headers, unclaimed_device):
    serial, priv = unclaimed_device

    ch = client.post("/device/challenge", json={"serial": serial}, headers=auth_headers)
    assert ch.status_code == 200, ch.text
    attest = client.post("/device/attest", headers=auth_headers, json={
        "instance_id": ch.json()["instance_id"],
        "signature": _sign(priv, ch.json(), serial),
    })
    assert attest.status_code == 200, attest.text

    # The device now shows up among the caller's owned serials.
    owned = client.get("/device/owned", headers=auth_headers)
    assert serial in owned.json()

    # Ownership can change at most once per cooldown window.
    again = client.post("/device/challenge", json={"serial": serial}, headers=auth_headers)
    assert again.status_code == 429


def test_attestation_bad_signature_400(client, auth_headers, unclaimed_device):
    serial, _ = unclaimed_device

    ch = client.post("/device/challenge", json={"serial": serial}, headers=auth_headers)
    assert ch.status_code == 200, ch.text
    bad = client.post("/device/attest", headers=auth_headers, json={
        "instance_id": ch.json()["instance_id"],
        "signature": base64.b64encode(b"\x00" * 70).decode(),
    })
    assert bad.status_code == 400

    # A failed attempt neither sets an owner nor consumes the cooldown window.
    with Session(engine) as session:
        device = session.get(Device, serial)
        assert device.owner_id is None
        assert device.last_attested_at is None
    assert client.post("/device/challenge", json={"serial": serial},
                       headers=auth_headers).status_code == 200
