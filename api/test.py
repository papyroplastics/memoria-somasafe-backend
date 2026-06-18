import numpy as np
from fastapi.testclient import TestClient
from ai_edge_litert.compiled_model import CompiledModel

from .main import app

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

    response = client.post(
        f"/model/quantize/feature-mlp/{model_id}",
        json={'parameters': parameters.tolist()},
    )
    assert response.status_code == 200
    assert len(response.content) > 0
