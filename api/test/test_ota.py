"""Tests for the /ota routes (api.routes.ota).

Firmware rows are inserted directly (random version strings, a random high
interface number) along with their images, so the tests don't depend on a
firmware image having been exported and seeded.
"""

import base64
import secrets
from datetime import timedelta

import pytest
from sqlmodel import select

from common.compression import compress, decompress
from common.db import Firmware, FirmwareImage, Session, engine, utcnow

SIGNATURE_HEADER = "X-Firmware-Signature"

BLOB_NEW = b"\xe9\x01\x02\x03-new-image"
BLOB_OLD = b"\xe9\x01\x02\x03-old-image"
SIGNATURE = b"not-a-real-der-signature"


@pytest.fixture
def firmwares():
    """Two throwaway signed builds on their own interface, the older one a day
    older. Deleting the rows on teardown takes their images with them."""
    interface = 1000 + secrets.randbelow(1_000_000)
    suffix = secrets.token_hex(4)
    new_version, old_version = f"9.9.1-{suffix}", f"9.9.0-{suffix}"
    builds = ((old_version, BLOB_OLD, [1], timedelta(days=1)),
              (new_version, BLOB_NEW, [1, 2], timedelta(0)))
    with Session(engine) as session:
        for version, blob, contracts, age in builds:
            firmware = Firmware(
                version=version, interface_version=interface,
                supported_contracts=contracts, size=len(blob),
                signature=SIGNATURE, created_at=utcnow() - age)
            session.add(firmware)
            session.flush()
            session.add(FirmwareImage(firmware_id=firmware.id,
                                      data=compress(blob)))
        session.commit()
    yield interface, new_version, old_version
    with Session(engine) as session:
        for fw in session.exec(
                select(Firmware).where(Firmware.interface_version == interface)):
            session.delete(fw)
        session.commit()


def test_versions_requires_auth(client, firmwares):
    interface, _, _ = firmwares
    assert client.get(f"/ota/versions/{interface}").status_code == 401


def test_versions_lists_interface_newest_first(client, auth_headers, firmwares):
    interface, new_version, old_version = firmwares
    resp = client.get(f"/ota/versions/{interface}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    versions = resp.json()
    assert [fw["version"] for fw in versions] == [new_version, old_version]
    assert versions[0]["supported_contracts"] == [1, 2]
    assert versions[0]["interface_version"] == interface
    assert versions[0]["size"] == len(BLOB_NEW)


def test_versions_unknown_interface_empty(client, auth_headers):
    resp = client.get("/ota/versions/999999999", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_download_requires_auth(client, firmwares):
    interface, new_version, _ = firmwares
    assert client.get(f"/ota/download/{interface}/{new_version}").status_code == 401


def test_download_requires_device_owner(client, deviceless_auth_headers, firmwares):
    interface, new_version, _ = firmwares
    resp = client.get(f"/ota/download/{interface}/{new_version}",
                      headers=deviceless_auth_headers)
    assert resp.status_code == 403


def test_download_serves_blob_and_signature(client, auth_headers, owned_device,
                                            firmwares):
    interface, new_version, _ = firmwares
    resp = client.get(f"/ota/download/{interface}/{new_version}",
                      headers=auth_headers)
    assert resp.status_code == 200, resp.text
    assert decompress(resp.content) == BLOB_NEW  # served zstd-compressed
    assert base64.b64decode(resp.headers[SIGNATURE_HEADER]) == SIGNATURE


def test_download_unknown_version_404(client, auth_headers, owned_device,
                                      firmwares):
    interface, _, _ = firmwares
    resp = client.get(f"/ota/download/{interface}/0.0.0-nope",
                      headers=auth_headers)
    assert resp.status_code == 404


def test_download_cooldown(client, auth_headers, owned_device, firmwares):
    interface, new_version, old_version = firmwares
    resp = client.get(f"/ota/download/{interface}/{new_version}",
                      headers=auth_headers)
    assert resp.status_code == 200
    # The cooldown is per interface: an immediate repeat is limited even for
    # another version of the same interface.
    resp = client.get(f"/ota/download/{interface}/{old_version}",
                      headers=auth_headers)
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1
