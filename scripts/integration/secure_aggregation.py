"""Headless secure-aggregation correctness harness — drives the real HTTP API end to end
like the secure path of ``scripts.fed_client``, but does no training: each client draws a
random weight tensor, submits the masked delta against the round's base ``W``, and the
script confirms the global weights the server bakes equal the plaintext mean of those
tensors, up to quantization and float32 error."""

import argparse
import base64

import numpy as np
from sqlmodel import Session

from common.celery_tasks import SECURE_AGG_TASK
from ml.model_list import MODELS
from common.config import SECURE_MIN_MEMBERS, SEED
from common.db import SubmissionType, engine, get_latest_version
from common.ratelimit import clear_model_limits
from common.secure_agg import (
    dequantize,
    generate_keypair,
    mask_vector,
    quantize,
    ring_sum,
)
from worker.celery_app import app

from scripts.common.api import (
    DEFAULT_BASE_URL,
    download_weights,
    get_descriptor,
    join,
    login,
    logout,
    submit_masked,
    wait_for_round,
)
from scripts.common.secure import seal_round


def run(base: str, key: str, clients: int, rounds: int) -> None:
    with Session(engine) as session:
        version = get_latest_version(session, key)
        if version is None:
            raise SystemExit(f"model '{key}' has no seeded version")
        if version.submission_type is not SubmissionType.secure:
            raise SystemExit(f"model '{key}' is '{version.submission_type.value}', not secure")

    if clients < SECURE_MIN_MEMBERS:
        raise SystemExit(f"--clients {clients} < SECURE_MIN_MEMBERS ({SECURE_MIN_MEMBERS})")

    keypairs = {f"test_{i}": generate_keypair() for i in range(1, clients + 1)}
    rng = np.random.default_rng(SEED)

    print(f"model={key} type=secure clients={clients} rounds={rounds} (no training)")

    for r in range(1, rounds + 1):
        prefix = f"round={r}/{rounds}"

        round_id = None
        seats = []
        for i in range(1, clients + 1):
            user = f"test_{i}"
            token = login(base, user, user)
            sk, pk = keypairs[user]
            resp = join(base, token, key, pk)
            round_id = resp["round_id"]
            seats.append((user, token, sk, resp["user_id"]))

        n = seal_round(round_id, SECURE_MIN_MEMBERS)
        print(f"{prefix} sealed round {round_id} with {n} members")

        desc = get_descriptor(base, seats[0][1], round_id)
        m, B, scale = desc["weight_count"], desc["clip_bound"], desc["scale"]
        clear_model_limits(key)
        raw, weights_id = download_weights(base, seats[0][1], key)
        if weights_id != desc["base_weights_id"]:
            raise SystemExit("served weights id != round base; client out of sync")
        base_weights = np.frombuffer(raw, dtype=np.float32)
        roster = [(e["user_id"], base64.b64decode(e["ka_public_key"]))
                  for e in desc["roster"]]

        masked_vecs, deltas = [], []
        for user, token, sk, my_id in seats:
            delta = rng.uniform(-B, B, m).astype(np.float32)
            q = quantize(delta, B, scale)
            y = mask_vector(q, my_id, roster, sk, round_id)
            submit_masked(base, token, round_id, y.astype("<u4").tobytes())
            logout(base, token)
            masked_vecs.append(y)
            deltas.append(delta)

        residual = float(np.max(np.abs(
            dequantize(ring_sum(masked_vecs), scale, n)
            - dequantize(ring_sum([quantize(d, B, scale) for d in deltas]), scale, n))))
        print(f"{prefix} mask-cancellation residual: {residual:.3e}")

        summary = wait_for_round(app.send_task(SECURE_AGG_TASK, args=[round_id]))
        print(f"{prefix} aggregated: {summary}")

        clear_model_limits(key)
        token = login(base, "test_1", "test_1")
        raw, _ = download_weights(base, token, key)
        logout(base, token)
        new_weights = np.frombuffer(raw, dtype=np.float32)

        expected = base_weights + np.mean(np.stack(deltas), axis=0).astype(np.float32)
        max_err = float(np.max(np.abs(new_weights - expected)))
        tol = 2.0 / scale + 1e-4
        verdict = "OK" if max_err < tol else "MISMATCH"
        print(f"{prefix} aggregate vs plaintext mean: max_err={max_err:.3e} "
              f"tol={tol:.3e} [{verdict}]")
        if max_err >= tol:
            raise SystemExit(f"{prefix} aggregation does not match the plaintext mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', default="cnn-ae", choices=sorted(MODELS),
                        help="secure-typed model to aggregate for")
    parser.add_argument("--clients", type=int, default=SECURE_MIN_MEMBERS,
                        help="cohort size, one test_N user each")
    parser.add_argument("--rounds", type=int, default=1, help="rounds to run")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="gateway base URL")
    args = parser.parse_args()

    run(args.base_url, args.model, args.clients, args.rounds)


if __name__ == "__main__":
    main()
