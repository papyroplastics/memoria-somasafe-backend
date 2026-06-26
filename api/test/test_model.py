"""Tests for the /model routes (api.routes.model).

These stop at the enqueue/poll boundary: no worker runs, so a submitted job
stays ``pending`` (202) and the int8 result is never produced (that path needs
TensorFlow and is covered by the worker, not here).
"""

import pytest

from common.config import QUANTIZE_DAILY_LIMIT

FINGERPRINT_HEADER = "X-Model-Fingerprint"
WEIGHTS_ID_HEADER = "X-Weights-ID"


def _model_with_weights(client, headers) -> str:
    resp = client.get("/model/list", headers=headers)
    assert resp.status_code == 200, resp.text
    for model in resp.json():
        if model["weights_version"] is not None:
            return model["key"]
    pytest.skip("no seeded model has weights; run the seed script first")


def test_list_requires_auth(client):
    assert client.get("/model/list").status_code == 401


def test_list_models(client, auth_headers):
    resp = client.get("/model/list", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_download_requires_device_owner(client, auth_headers):
    key = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/download/trainable/{key}", headers=auth_headers)
    assert resp.status_code == 403


def test_download_echoes_fingerprint(client, auth_headers, owned_device):
    key = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/download/trainable/{key}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers[FINGERPRINT_HEADER]


def test_download_cooldown(client, auth_headers, owned_device):
    key = _model_with_weights(client, auth_headers)
    assert client.get(f"/model/download/trainable/{key}",
                      headers=auth_headers).status_code == 200
    # An immediate repeat on the same model is rate limited.
    assert client.get(f"/model/download/trainable/{key}",
                      headers=auth_headers).status_code == 429


def test_weights_carries_id_and_fingerprint(client, auth_headers, owned_device):
    key = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/weights/{key}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers[FINGERPRINT_HEADER]
    assert int(resp.headers[WEIGHTS_ID_HEADER]) > 0


def test_quantize_enqueues_and_polls_pending(client, auth_headers, owned_device):
    key = _model_with_weights(client, auth_headers)
    weights_id = int(client.get(f"/model/weights/{key}",
                                headers=auth_headers).headers[WEIGHTS_ID_HEADER])

    submit = client.post(f"/model/quantize/{key}", headers=auth_headers,
                         json={"parameters": [0.0, 1.0], "weights_id": weights_id})
    assert submit.status_code == 202, submit.text
    job_id = submit.json()["job_id"]

    # No worker consumes the queue, so the result endpoint reports it pending.
    result = client.get(f"/model/quantize/result/{job_id}", headers=auth_headers)
    assert result.status_code == 202
    assert result.json()["status"] == "pending"


def test_quantize_unknown_weights_400(client, auth_headers, owned_device):
    key = _model_with_weights(client, auth_headers)
    resp = client.post(f"/model/quantize/{key}", headers=auth_headers,
                       json={"parameters": [0.0], "weights_id": 999_999_999})
    assert resp.status_code == 400


def test_quantize_daily_limit(client, auth_headers, owned_device):
    key = _model_with_weights(client, auth_headers)
    weights_id = int(client.get(f"/model/weights/{key}",
                                headers=auth_headers).headers[WEIGHTS_ID_HEADER])
    body = {"parameters": [0.0], "weights_id": weights_id}

    for _ in range(QUANTIZE_DAILY_LIMIT):
        assert client.post(f"/model/quantize/{key}", headers=auth_headers,
                           json=body).status_code == 202
    # One over the daily cap is rejected.
    assert client.post(f"/model/quantize/{key}", headers=auth_headers,
                       json=body).status_code == 429
