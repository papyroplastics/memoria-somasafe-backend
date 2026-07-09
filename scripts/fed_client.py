"""Headless federated client harness — drives the real HTTP API end to end.

For each round, and for each training subject (as user ``test_N``), it:
  1. logs in (``POST /auth/token``),
  2. pulls the current global trainable artifact (``GET /model/download/trainable``),
  3. trains one pass on that subject through the on-device LiteRT ``CompiledModel``
     interface (the same runtime the phone uses),
  4. uploads the resulting parameters to the model's submission endpoint,
  5. logs out.

After every client has submitted it queues one aggregation round and polls
``GlobalWeights`` until the new snapshot appears, then scores it on the held-out
subjects. A per-round metric series is written out for the convergence curve.

    uv run -m scripts.fed_client
    uv run -m scripts.fed_client --model cnn-ae --rounds 5 --eval-subjects 2

Prereqs: the services are up (api, worker, redis, postgres), the DB was seeded
with ``--test-users``, and the model is trained/seeded. There are no local epochs
— the system takes a single submission per client per round.
"""

import argparse
import json
import time
from typing import cast
from pathlib import Path

import numpy as np
import requests
from ai_edge_litert.compiled_model import CompiledModel
from sqlmodel import Session, select

from common.config import DATASETS_DIR
from common.db import (
    GlobalWeights,
    SubmissionType,
    engine,
    get_latest_version,
)
from ml.model_list import MODELS
from worker.celery_app import app

AGGREGATION_TASK = "worker.tasks.federated_aggregation"
WEIGHTS_ID_HEADER = "X-Weights-ID"
DEFAULT_BASE_URL = "http://localhost:8000"


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

    def train_pass(self, dataset) -> float:
        total, batches = 0.0, 0
        for batch in dataset:
            arrays = [t.numpy() for t in batch]
            total += float(self._run("train", arrays)["loss"].reshape(-1)[0])
            batches += 1
        return total / batches if batches else 0.0

    def parameters(self) -> np.ndarray:
        return self._run("save", [])["parameters"].astype(np.float32)

    def eval(self, datapoint) -> dict[str, np.ndarray]:
        """Run the eval signature on one dataset batch, returning the output tensors
        keyed by output name. Extra datapoint tensors (targets the eval signature
        doesn't take, like the MLP's labels) are matched out by ``_run`` by name."""
        return self._run("eval", [t.numpy() for t in datapoint])


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def login(base: str, username: str, password: str) -> str:
    resp = requests.post(f"{base}/auth/token",
                         data={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_trainable(base: str, token: str, key: str) -> tuple[bytes, int]:
    resp = requests.get(f"{base}/model/download/trainable/{key}", headers=_auth(token))
    resp.raise_for_status()
    return resp.content, int(resp.headers[WEIGHTS_ID_HEADER])


def submit(base: str, token: str, key: str, weights_id: int, body: bytes,
           submission_type: SubmissionType) -> None:
    path = "quantize" if submission_type is SubmissionType.quantize else "raw"
    resp = requests.post(
        f"{base}/model/submit/{path}/{key}/{weights_id}",
        headers=_auth(token) | {"Content-Type": "application/octet-stream"},
        data=body,
    )
    resp.raise_for_status()


def logout(base: str, token: str) -> None:
    requests.post(f"{base}/auth/logout", headers=_auth(token))


def _newest_weights(session: Session, version_id: int) -> GlobalWeights | None:
    return session.exec(
        select(GlobalWeights)
        .where(GlobalWeights.version_id == version_id)
        .order_by(GlobalWeights.created_at.desc())  # type: ignore[attr-defined]
    ).first()


def wait_for_aggregation(version_id: int, previous_id: int | None,
                         timeout: float = 10.0, interval: float = 0.1) -> GlobalWeights:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with Session(engine) as session:
            newest = _newest_weights(session, version_id)
            if newest is not None and newest.id != previous_id:
                if not newest.valid:
                    raise SystemExit("aggregation produced an invalid round "
                                     "(artifact export failed)")
                return newest
        time.sleep(interval)
    raise SystemExit(f"aggregation did not produce new weights within {timeout}s")


def run(base: str, key: str, rounds: int, eval_subjects: int, out: Path) -> None:
    spec = MODELS[key]
    trainer = spec.build_trainer(DATASETS_DIR)
    subject_datasets, _ = trainer.subject_datasets(DATASETS_DIR, train_split=1.0)

    if eval_subjects >= len(subject_datasets):
        raise SystemExit(f"--eval-subjects {eval_subjects} leaves no training subjects "
                         f"({len(subject_datasets)} available)")
    if eval_subjects > 0:
        client_datasets = subject_datasets[:-eval_subjects]
        # Materialize each held-out subject and concatenate at the Python level, so the
        # already-cached subject datasets are never re-combined into a new tf pipeline.
        eval_data = [dp for ds in subject_datasets[-eval_subjects:] for dp in list(ds)]
    else:
        client_datasets = subject_datasets
        eval_data = None

    print(f"model={key} type={spec.submission_type.value} clients={len(client_datasets)} "
          f"eval_subjects={eval_subjects} rounds={rounds}")

    history: list[dict] = []

    def score(client: LiteRTClient, round_idx: int) -> None:
        if not eval_data:
            return
        outputs = [client.eval(dp) for dp in eval_data]
        value = trainer.eval_metrics(eval_data, outputs)[trainer.primary_metric]
        history.append({"round": round_idx, trainer.primary_metric: value})
        print(f"round={round_idx} {trainer.primary_metric}={value:.6f}")

    for r in range(1, rounds + 1):
        version_id = 0
        with Session(engine) as session:
            version = get_latest_version(session, key)
            if version is None:
                raise SystemExit(f"model '{key}' has no seeded version")
            version_id = cast(int, version.id)
            previous_id = (n.id if (n := _newest_weights(session, version_id)) else None)

        round_global: bytes | None = None
        for i, dataset in enumerate(client_datasets, start=1):
            user = f"test_{i}"
            token = login(base, user, user)
            artifact, weights_id = download_trainable(base, token, key)
            client = LiteRTClient(artifact, trainer.dataset_tensors)
            if round_global is None:
                round_global = artifact
                score(client, r - 1)  # global weights this round trained from

            client.train_pass(dataset)
            submit(base, token, key, weights_id, client.parameters().tobytes(),
                   spec.submission_type)
            logout(base, token)

        app.send_task(AGGREGATION_TASK, args=[key])
        wait_for_aggregation(version_id, previous_id)
        print(f"round={r} aggregated")

    # Final global weights, after the last round's aggregation.
    token = login(base, "test_1", "test_1")
    artifact, _ = download_trainable(base, token, key)
    logout(base, token)

    client = LiteRTClient(artifact, trainer.dataset_tensors)
    score(client, rounds)

    out.write_text(json.dumps(history, indent=2))
    print(f"wrote {len(history)} metrics to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="cnn-ae", choices=sorted(MODELS),
                        help="model to run the loop for (default: cnn-ae)")
    parser.add_argument("--rounds", type=int, default=5, help="global rounds (default: 5)")
    parser.add_argument("--eval-subjects", type=int, default=2,
                        help="subjects reserved from the end for evaluation (default: 2)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"gateway base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--out", type=Path, default=None,
                        help="metrics JSON output path (default: <model>-fed-history.json)")
    args = parser.parse_args()

    out = args.out or Path(f"{args.model}-fed-history.json")
    run(args.base_url, args.model, args.rounds, args.eval_subjects, out)


if __name__ == "__main__":
    main()
