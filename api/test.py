import os
import tempfile

# Configure a throwaway SQLite DB and an eager (broker-less) Celery before the
# app modules read these at import time.
os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{tempfile.gettempdir()}/somasafe_test.db")
os.environ.setdefault("BROKER_URL", "memory://")

import numpy as np
from fastapi.testclient import TestClient
from ai_edge_litert.compiled_model import CompiledModel

from worker.celery_app import app as celery_app

celery_app.conf.task_always_eager = True
import worker.tasks  # noqa: F401,E402 - registers quantize_submission for eager run

from common.db import init_db
from api.main import app

init_db()
client = TestClient(app)


def test_quantize_feature_mlp():
    list_response = client.get("/model/list")
    assert list_response.status_code == 200
    model_info = next(m for m in list_response.json() if m["key"] == "feature-mlp")
    model_id = model_info["model_id"]

    response = client.get(f"/model/trainable/feature-mlp/{model_id}")
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
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    # Poll the result endpoint for the quantized .tflite.
    result = client.get(f"/model/quantize/result/{job_id}")
    assert result.status_code == 200
    assert len(result.content) > 0
