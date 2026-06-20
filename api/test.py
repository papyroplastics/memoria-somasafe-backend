import os
import tempfile

# Configure a throwaway SQLite DB and an eager (broker-less) Celery before the
# app modules read these at import time.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{tempfile.gettempdir()}/somasafe_test.db")
os.environ.setdefault("BROKER_URL", "memory://")

import fakeredis
import numpy as np
from fastapi.testclient import TestClient
from ai_edge_litert.compiled_model import CompiledModel

from worker.celery_app import app as celery_app

celery_app.conf.task_always_eager = True
import worker.tasks  # noqa: F401,E402 - registers quantize_submission for eager run

import common.ratelimit
from common.db import Session, User, engine, init_db
from api.auth import hash_password
from api.main import app
from scripts.seed import seed_models

# Rate limiting talks to Redis; swap in an in-memory fake for the test.
common.ratelimit._client = fakeredis.FakeStrictRedis()

init_db()

TEST_USER = "tester"
TEST_PASSWORD = "testpass"

with Session(engine) as _session:
    seed_models(_session)
    if _session.get(User, 1) is None:
        _session.add(User(username=TEST_USER, hashed_password=hash_password(TEST_PASSWORD)))
        _session.commit()

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
    model_id = model_info["model_id"]

    response = client.get(f"/model/trainable/feature-mlp/{model_id}", headers=headers)
    assert response.status_code == 200

    compiled = CompiledModel.from_buffer(response.content)
    out_buf = compiled.create_output_buffer_by_name('save', 'parameters')
    num_elements = int(np.prod(out_buf.get_tensor_details()['shape']))
    compiled.run_by_name('save', {}, {'parameters': out_buf})
    parameters = out_buf.read(num_elements, np.float32)

    # Submit weights -> 202 + job id; with eager Celery the worker runs inline.
    submit = client.post(
        f"/model/quantize/feature-mlp/{model_id}",
        json={'parameters': parameters.tolist()},
        headers=headers,
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    # Poll the result endpoint for the quantized .tflite.
    result = client.get(f"/model/quantize/result/{job_id}", headers=headers)
    assert result.status_code == 200
    assert len(result.content) > 0
