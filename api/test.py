import numpy as np
from fastapi.testclient import TestClient
from ai_edge_litert.compiled_model import CompiledModel

from .main import app

client = TestClient(app)

def test_quantize_feature_mlp():
    response = client.get("/model/trainable/feature-mlp")
    assert response.status_code == 200

    compiled = CompiledModel.from_buffer(response.content)

    out_buf = compiled.create_output_buffer_by_name('save', 'parameters')
    num_elements = int(np.prod(out_buf.get_tensor_details()['shape']))
    compiled.run_by_name('save', {}, {'parameters': out_buf})
    parameters = out_buf.read(num_elements, np.float32)

    response = client.post(
        "/model/quantize/feature-mlp",
        json={'parameters': parameters.tolist()},
    )
    assert response.status_code == 200
    assert len(response.content) > 0
