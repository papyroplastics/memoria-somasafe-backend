"""Tests for the /model routes (api.routes.model).

These stop at the enqueue/poll boundary: no worker runs, so a submitted job
stays ``pending`` (202) and the int8 result is never produced (that path needs
TensorFlow and is covered by the worker, not here).
"""

import pytest

from common.config import QUANTIZE_DAILY_LIMIT, SUBMIT_DAILY_LIMIT

FINGERPRINT_HEADER = "X-Model-Fingerprint"
MODEL_VERSION_HEADER = "X-Model-Version"
WEIGHTS_ID_HEADER = "X-Weights-ID"

OCTET_STREAM = {"Content-Type": "application/octet-stream"}


def _model_with_weights(client, headers) -> dict:
    resp = client.get("/model/list", headers=headers)
    assert resp.status_code == 200, resp.text
    for model in resp.json():
        if model["weights_version"] is not None:
            return model
    pytest.skip("no seeded model has weights; run the seed script first")


def _download_trainable(client, headers, key):
    resp = client.get(f"/model/download/trainable/{key}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp


def _zero_params(model: dict) -> bytes:
    return b"\x00" * (model["param_count"] * 4)


def test_list_requires_auth(client):
    assert client.get("/model/list").status_code == 401


def test_list_models(client, auth_headers):
    resp = client.get("/model/list", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_versions_lists_latest(client, auth_headers):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/versions/{model['key']}", headers=auth_headers)
    assert resp.status_code == 200
    versions = resp.json()
    assert versions and versions[0]["version"] == model["version"]


def test_download_requires_device_owner(client, auth_headers):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/download/trainable/{model['key']}", headers=auth_headers)
    assert resp.status_code == 403


def test_download_echoes_version_headers(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    assert resp.headers[FINGERPRINT_HEADER] == model["fingerprint"]
    assert int(resp.headers[MODEL_VERSION_HEADER]) == model["version"]
    assert int(resp.headers[WEIGHTS_ID_HEADER]) > 0


def test_download_cooldown(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    _download_trainable(client, auth_headers, model["key"])
    # An immediate repeat on the same model is rate limited.
    assert client.get(f"/model/download/trainable/{model['key']}",
                      headers=auth_headers).status_code == 429


def test_download_unknown_version_404(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/download/trainable/{model['key']}?version=999",
                      headers=auth_headers)
    assert resp.status_code == 404


def test_quantize_enqueues_and_polls_pending(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])

    submit = client.post(f"/model/quantize/submit/{model['key']}/{weights_id}",
                         headers=auth_headers | OCTET_STREAM,
                         content=_zero_params(model))
    assert submit.status_code == 202, submit.text
    job_id = submit.json()["job_id"]

    # No worker consumes the queue, so the result endpoint reports it pending.
    result = client.get(f"/model/quantize/result/{job_id}", headers=auth_headers)
    assert result.status_code == 202
    assert result.json()["status"] == "pending"


def test_quantize_unknown_weights_400(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = client.post(f"/model/quantize/submit/{model['key']}/999999999",
                       headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model))
    assert resp.status_code == 400


def test_quantize_wrong_length_400(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    resp = client.post(f"/model/quantize/submit/{model['key']}/{weights_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=b"\x00" * 8)
    assert resp.status_code == 400


def test_quantize_daily_limit(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    url = f"/model/quantize/submit/{model['key']}/{weights_id}"

    for _ in range(QUANTIZE_DAILY_LIMIT):
        assert client.post(url, headers=auth_headers | OCTET_STREAM,
                           content=_zero_params(model)).status_code == 202
    # One over the daily cap is rejected.
    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model)).status_code == 429


def test_submit_only_accepts(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])

    resp = client.post(f"/model/submit/{model['key']}/{weights_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model))
    assert resp.status_code == 202, resp.text
    assert resp.json()["submission_id"] > 0


def test_submit_daily_limit(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    url = f"/model/submit/{model['key']}/{weights_id}"

    for _ in range(SUBMIT_DAILY_LIMIT):
        assert client.post(url, headers=auth_headers | OCTET_STREAM,
                           content=_zero_params(model)).status_code == 202
    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model)).status_code == 429
