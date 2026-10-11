"""Tests for the /model routes (api.routes.model). No worker runs, so quantize jobs
stay pending."""

import pytest
from sqlmodel import select

from common.celery_tasks import FED_AGG_TASK
from common.compression import decompress
from common.config import SUBMIT_DAILY_LIMIT
from common.db import (
    ClientDeltaSubmission,
    GlobalWeights,
    ModelVersion,
    Session,
    count_round_submissions,
    engine,
    get_latest_weights,
)
from ..routes import model as model_routes

FINGERPRINT_HEADER = "X-Model-Fingerprint"
MODEL_VERSION_HEADER = "X-Model-Version"
WEIGHTS_ID_HEADER = "X-Weights-ID"

OCTET_STREAM = {"Content-Type": "application/octet-stream"}


def _model_with_weights(client, headers, submission_type: str | None = None,
                        exclude_submission_type: str | None = None) -> dict:
    resp = client.get("/model/list", headers=headers)
    assert resp.status_code == 200, resp.text
    for model in resp.json():
        if model["weights_version"] is None:
            continue
        if submission_type is not None and model["submission_type"] != submission_type:
            continue
        if exclude_submission_type is not None and \
                model["submission_type"] == exclude_submission_type:
            continue
        return model
    pytest.skip(f"no seeded {submission_type or ''} model has weights; "
                f"run the seed script first")


def _download_trainable(client, headers, key):
    resp = client.get(f"/model/download/trainable/{key}", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp


def _zero_params(model: dict) -> bytes:
    return b"\x00" * (model["weight_count"] * 4)


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


def test_download_requires_device_owner(client, auth_headers, deviceless_auth_headers):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/download/trainable/{model['key']}",
                      headers=deviceless_auth_headers)
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


def test_weights_requires_device_owner(client, auth_headers, deviceless_auth_headers):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/weights/{model['key']}",
                      headers=deviceless_auth_headers)
    assert resp.status_code == 403


def test_weights_serves_flat_buffer(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/weights/{model['key']}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    # The body decompresses to exactly the model's flat float32 weight buffer,
    # and echoes the same version/weights headers the artifact download does.
    assert len(decompress(resp.content)) == model["weight_count"] * 4
    assert resp.headers[FINGERPRINT_HEADER] == model["fingerprint"]
    assert int(resp.headers[MODEL_VERSION_HEADER]) == model["version"]
    assert int(resp.headers[WEIGHTS_ID_HEADER]) > 0


def test_weights_shares_base_with_artifact(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    trainable = _download_trainable(client, auth_headers, model["key"])
    weights = client.get(f"/model/weights/{model['key']}", headers=auth_headers)
    assert weights.status_code == 200, weights.text
    assert weights.headers[WEIGHTS_ID_HEADER] == trainable.headers[WEIGHTS_ID_HEADER]


def test_weights_cooldown(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    assert client.get(f"/model/weights/{model['key']}",
                      headers=auth_headers).status_code == 200
    # An immediate repeat on the same model is rate limited.
    assert client.get(f"/model/weights/{model['key']}",
                      headers=auth_headers).status_code == 429


def test_weights_download_slots_are_independent(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    _download_trainable(client, auth_headers, model["key"])
    assert client.get(f"/model/weights/{model['key']}",
                      headers=auth_headers).status_code == 200


def test_weights_unknown_version_404(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = client.get(f"/model/weights/{model['key']}?version=999",
                      headers=auth_headers)
    assert resp.status_code == 404


def test_quantize_enqueues_and_polls_pending(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers, "quantize")
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])

    submit = client.post(f"/model/submit/quantize/{model['key']}/{weights_id}",
                         headers=auth_headers | OCTET_STREAM,
                         content=_zero_params(model))
    assert submit.status_code == 202, submit.text
    job_id = submit.json()["job_id"]

    result = client.get(f"/model/quantize/result/{job_id}", headers=auth_headers)
    assert result.status_code == 202, result.text
    assert result.json()["status"] == "pending"


def test_quantize_unknown_weights_400(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers, "quantize")
    resp = client.post(f"/model/submit/quantize/{model['key']}/999999999",
                       headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model))
    assert resp.status_code == 400


def test_quantize_wrong_length_400(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers, "quantize")
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    resp = client.post(f"/model/submit/quantize/{model['key']}/{weights_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=b"\x00" * 8)
    assert resp.status_code == 400


def test_quantize_daily_limit(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers, "quantize")
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    url = f"/model/submit/quantize/{model['key']}/{weights_id}"

    for _ in range(SUBMIT_DAILY_LIMIT):
        assert client.post(url, headers=auth_headers | OCTET_STREAM,
                           content=_zero_params(model)).status_code == 202
    # One over the daily cap is rejected.
    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model)).status_code == 429


def test_quantize_rejects_raw_model(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers, exclude_submission_type="quantize")
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    resp = client.post(f"/model/submit/quantize/{model['key']}/{weights_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model))
    assert resp.status_code == 404


def test_submit_only_accepts(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])

    resp = client.post(f"/model/submit/raw/{model['key']}/{weights_id}",
                       headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model))
    assert resp.status_code == 202, resp.text
    assert resp.json()["submission_id"] > 0


def test_submit_daily_limit(client, auth_headers, owned_device):
    model = _model_with_weights(client, auth_headers)
    resp = _download_trainable(client, auth_headers, model["key"])
    weights_id = int(resp.headers[WEIGHTS_ID_HEADER])
    url = f"/model/submit/raw/{model['key']}/{weights_id}"

    for _ in range(SUBMIT_DAILY_LIMIT):
        assert client.post(url, headers=auth_headers | OCTET_STREAM,
                           content=_zero_params(model)).status_code == 202
    assert client.post(url, headers=auth_headers | OCTET_STREAM,
                       content=_zero_params(model)).status_code == 429


def _user_headers(client, i: int) -> dict:
    resp = client.post("/auth/token", data={"username": f"test_{i}", "password": f"test_{i}"})
    if resp.status_code != 200:
        pytest.skip("seed the test users first (make db-seed)")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def raw_round(client, auth_headers):
    model = _model_with_weights(client, auth_headers, "raw")
    with Session(engine) as session:
        base = get_latest_weights(session, model["key"])
        version = session.get(ModelVersion, base.version_id)
        limit, base_id, version_id = version.round_limit, base.id, version.id
        floor = session.exec(select(ClientDeltaSubmission.id)
                             .order_by(ClientDeltaSubmission.id.desc())).first() or 0

    def set_limit(value: int) -> None:
        with Session(engine) as session:
            version = session.get(ModelVersion, version_id)
            version.round_limit = value
            session.add(version)
            session.commit()

    def submissions() -> int:
        with Session(engine) as session:
            return count_round_submissions(session, base_id)

    yield model, base_id, set_limit, submissions
    set_limit(limit)
    with Session(engine) as session:
        session.get(GlobalWeights, base_id).closed = False
        for row in session.exec(select(ClientDeltaSubmission)
                                .where(ClientDeltaSubmission.id > floor)):
            session.delete(row)
        session.commit()


def _submit_raw(client, headers, model, base_id):
    return client.post(f"/model/submit/raw/{model['key']}/{base_id}",
                       headers=headers | OCTET_STREAM, content=_zero_params(model))


def test_full_round_rejects_new_submitters(client, raw_round):
    model, base_id, set_limit, submissions = raw_round
    first = _user_headers(client, 1)
    assert _submit_raw(client, first, model, base_id).status_code == 202
    set_limit(submissions())

    assert _submit_raw(client, _user_headers(client, 2), model, base_id).status_code == 409
    assert _submit_raw(client, first, model, base_id).status_code == 202


def test_filling_round_dispatches_aggregation(client, raw_round, monkeypatch):
    model, base_id, set_limit, submissions = raw_round
    sent = []
    monkeypatch.setattr(model_routes.celery_app, "send_task",
                        lambda name, args=None, **_: sent.append((name, args)))
    set_limit(submissions() + 2)

    assert _submit_raw(client, _user_headers(client, 1), model, base_id).status_code == 202
    assert sent == []
    assert _submit_raw(client, _user_headers(client, 2), model, base_id).status_code == 202
    assert sent == [(FED_AGG_TASK, [model["key"], base_id])]

    late = _submit_raw(client, _user_headers(client, 1), model, base_id)
    assert late.status_code == 409 and "closed" in late.json()["detail"]
    assert len(sent) == 1
