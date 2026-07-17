"""On-device LiteRT training/eval driver shared by the client harnesses.

Runs a trainable ``.tflite`` through LiteRT's ``CompiledModel`` exactly as the
phone does, so a headless harness exercises the same runtime as the device.
"""

import numpy as np
from ai_edge_litert import schema_py_generated as tflite_schema
from ai_edge_litert.compiled_model import CompiledModel
from tqdm import tqdm

from common.config import DISABLE_TQDM


def _tensor_quantization(tflite_bytes: bytes, name: str) -> tuple[float, int]:
    """Per-tensor (scale, zero_point) for a named tensor, read straight from the
    flatbuffer schema: CompiledModel's tensor-detail dicts no longer carry a
    'quantization' key on this LiteRT version (name/index/dtype/shape only)."""
    model = tflite_schema.Model.GetRootAsModel(tflite_bytes, 0)
    for i in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(i)
        for j in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(j)
            if tensor.Name().decode() == name:
                q = tensor.Quantization()
                return float(q.Scale(0)), int(q.ZeroPoint(0))
    raise ValueError(f"tensor '{name}' not found in any subgraph")

class LiteRTClient:
    """Trains and evaluates a trainable ``.tflite`` through LiteRT's CompiledModel,
    driving the model's ``train`` / ``save`` / ``eval`` signatures exactly as the
    on-device client does. Weight updates accumulate in the compiled model's
    resource variables across ``train`` calls; ``save`` reads them back out."""

    def __init__(self, tflite_bytes: bytes, tensor_names: list[str]):
        self.model = CompiledModel.from_buffer(tflite_bytes)
        self.signatures = self.model.get_signature_list()
        self.tensor_names = tensor_names

    def _run(self, signature: str, arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
        named = dict(zip(self.tensor_names, arrays))
        input_map = {}
        for name in self.signatures[signature]["inputs"]:
            if name not in named:
                raise ValueError(f"no input array named '{name}' for signature '{signature}'")
            buffer = self.model.create_input_buffer_by_name(signature, name)
            buffer.write(np.ascontiguousarray(named[name], dtype=np.float32))
            input_map[name] = buffer

        output_details = self.model.get_output_tensor_details(signature)
        output_map = {
            name: self.model.create_output_buffer_by_name(signature, name)
            for name in self.signatures[signature]["outputs"]
        }
        self.model.run_by_name(signature, input_map, output_map)

        out = {}
        for name, buffer in output_map.items():
            shape = output_details[name]["shape"]
            count = int(np.prod(shape)) if len(shape) else 1
            out[name] = buffer.read(count, np.float32)
        return out

    def train_pass(self, dataset, prefix: str = "") -> float:
        total, batches = 0.0, 0
        for batch in tqdm(dataset, desc=f"{prefix} train".strip(),
                          leave=False, disable=DISABLE_TQDM):
            arrays = [t.numpy() for t in batch]
            total += float(self._run("train", arrays)["loss"].reshape(-1)[0])
            batches += 1
        return total / batches if batches else 0.0

    def weights(self) -> np.ndarray:
        return self._run("save", [])["weights"].astype(np.float32)

    def restore(self, weights: np.ndarray) -> None:
        """Load a flat global-weights buffer into the compiled model's resource
        variables via the ``restore`` signature, so a fresh snapshot pulled from
        ``/model/weights`` can be applied without rebuilding the model from a new
        trainable artifact."""
        [name] = self.signatures["restore"]["inputs"]
        buffer = self.model.create_input_buffer_by_name("restore", name)
        buffer.write(np.ascontiguousarray(weights, dtype=np.float32))
        output_map = {
            out: self.model.create_output_buffer_by_name("restore", out)
            for out in self.signatures["restore"]["outputs"]
        }
        self.model.run_by_name("restore", {name: buffer}, output_map)

    def eval(self, datapoint) -> dict[str, np.ndarray]:
        """Run the eval signature on one dataset batch, returning the output tensors
        keyed by output name. Extra datapoint tensors (targets the eval signature
        doesn't take, like the MLP's labels) are matched out by ``_run`` by name."""
        return self._run("eval", [t.numpy() for t in datapoint])


def infer_int8(tflite_bytes: bytes, X_norm: np.ndarray, signature: str = "infer") -> np.ndarray:
    """Runs a quantized int8 model's single-input/single-output signature over
    per-row logits, quantizing/dequantizing with the tensors' own scale/zero-point
    exactly as the on-device int8 runtime does."""
    model = CompiledModel.from_buffer(tflite_bytes)
    sig = model.get_signature_list()[signature]
    in_name, out_name = sig["inputs"][0], sig["outputs"][0]
    in_details = model.get_input_tensor_details(signature)[in_name]
    out_details = model.get_output_tensor_details(signature)[out_name]
    iscale, izp = _tensor_quantization(tflite_bytes, in_details["name"])
    oscale, ozp = _tensor_quantization(tflite_bytes, out_details["name"])

    out = np.empty(len(X_norm), dtype=np.float32)
    for i, x in enumerate(X_norm):
        q = np.clip(np.round(x / iscale + izp), -128, 127).astype(np.int8).reshape(in_details["shape"])
        input_buffer = model.create_input_buffer_by_name(signature, in_name)
        input_buffer.write(q)
        output_buffer = model.create_output_buffer_by_name(signature, out_name)
        model.run_by_name(signature, {in_name: input_buffer}, {out_name: output_buffer})
        o = float(output_buffer.read(1, np.int8)[0])
        out[i] = (o - ozp) * oscale
    return out
