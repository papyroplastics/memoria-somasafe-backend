"""On-device LiteRT training/eval driver shared by the client harnesses.

Runs a trainable ``.tflite`` through LiteRT's ``CompiledModel`` exactly as the
phone does, so a headless harness exercises the same runtime as the device.
"""

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from tqdm import tqdm


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
        for batch in tqdm(dataset, desc=f"{prefix} train".strip(), leave=False):
            arrays = [t.numpy() for t in batch]
            total += float(self._run("train", arrays)["loss"].reshape(-1)[0])
            batches += 1
        return total / batches if batches else 0.0

    def weights(self) -> np.ndarray:
        return self._run("save", [])["weights"].astype(np.float32)

    def eval(self, datapoint) -> dict[str, np.ndarray]:
        """Run the eval signature on one dataset batch, returning the output tensors
        keyed by output name. Extra datapoint tensors (targets the eval signature
        doesn't take, like the MLP's labels) are matched out by ``_run`` by name."""
        return self._run("eval", [t.numpy() for t in datapoint])
