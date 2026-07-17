"""HTTP helpers for the headless client harnesses — auth, artifact download, and
the dense + secure submission/aggregation endpoints. TensorFlow-free, so the
aggregation-only harness (which never trains) can share them too.
"""

import base64

import requests

from common.db import SubmissionType
from common.compression import decompress

DEFAULT_BASE_URL = "http://localhost:8000"
WEIGHTS_ID_HEADER = "X-Weights-ID"
OCTET_STREAM = {"Content-Type": "application/octet-stream"}


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def login(base: str, username: str, password: str) -> str:
    resp = requests.post(f"{base}/auth/token",
                         data={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json()["access_token"]


def logout(base: str, token: str) -> None:
    requests.post(f"{base}/auth/logout", headers=auth(token))


def download_trainable(base: str, token: str, key: str) -> tuple[bytes, int]:
    resp = requests.get(f"{base}/model/download/trainable/{key}", headers=auth(token))
    resp.raise_for_status()
    return decompress(resp.content), int(resp.headers[WEIGHTS_ID_HEADER])


def download_weights(base: str, token: str, key: str) -> tuple[bytes, int]:
    resp = requests.get(f"{base}/model/weights/{key}", headers=auth(token))
    resp.raise_for_status()
    return decompress(resp.content), int(resp.headers[WEIGHTS_ID_HEADER])


def submit_delta(base: str, token: str, key: str, weights_id: int, body: bytes,
                 submission_type: SubmissionType) -> None:
    path = "quantize" if submission_type is SubmissionType.quantize else "raw"
    resp = requests.post(
        f"{base}/model/submit/{path}/{key}/{weights_id}",
        headers=auth(token) | OCTET_STREAM, data=body,
    )
    resp.raise_for_status()


def join(base: str, token: str, key: str, ka_public_key: bytes) -> dict:
    resp = requests.post(
        f"{base}/model/secure/join/{key}", headers=auth(token),
        json={"ka_public_key": base64.b64encode(ka_public_key).decode()},
    )
    resp.raise_for_status()
    return resp.json()


def get_descriptor(base: str, token: str, round_id: int) -> dict:
    resp = requests.get(f"{base}/model/secure/round/{round_id}", headers=auth(token))
    resp.raise_for_status()
    return resp.json()


def submit_masked(base: str, token: str, round_id: int, body: bytes) -> None:
    resp = requests.post(
        f"{base}/model/secure/submit/{round_id}",
        headers=auth(token) | OCTET_STREAM, data=body,
    )
    resp.raise_for_status()


def wait_for_aggregation(result, key: str, timeout: float = 300.0) -> str:
    """Block on a dense ``federated_aggregation`` task and return its summary for
    ``key``, raising if the round was skipped or its export invalidated it."""
    message = result.get(timeout=timeout).get(key, "no summary returned")
    if message.startswith("skipped") or "export failed" in message:
        raise SystemExit(f"aggregation for {key} produced no new weights: {message}")
    return message


def wait_for_round(result, timeout: float = 300.0) -> str:
    """Block on a ``secure_aggregation`` task and return its summary, raising if the
    round was skipped/failed or its export invalidated it."""
    summary = result.get(timeout=timeout)
    if summary.startswith(("skipped", "failed")) or "export failed" in summary:
        raise SystemExit(f"secure round produced no new weights: {summary}")
    return summary
