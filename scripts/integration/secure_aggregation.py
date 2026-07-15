"""Headless secure-aggregation *correctness* harness — drives the real HTTP API
end to end like the secure path of ``scripts.fed_client``, but does **no training**.

The point is to check the masking + summation pipeline in isolation: instead of
training a model, each client draws a random weight tensor, submits the masked
delta against the round's base ``W``, and the script confirms the global weights
the server bakes equal the plaintext mean of those tensors (up to quantization
and float32 error). With no train pass it runs fast and independently of whether a
model is trained — only that a ``secure`` version and its base weights are seeded.

    uv run -m scripts.secure_aggregation --clients 4 --rounds 3

Prereqs: services up (api, worker, redis, postgres), DB seeded with ``--test-users``
(one ``test_N`` per client), and a ``secure`` model version with base weights.
"""

import argparse
import base64

import numpy as np
from sqlmodel import Session

from common.celery_tasks import SECURE_AGG_TASK
from ml.model_list import MODELS
from common.config import SECURE_MIN_MEMBERS, SEED
from common.db import (
    GlobalWeights,
    SubmissionType,
    engine,
    get_latest_version,
    get_latest_weights,
)
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
    get_descriptor,
    join,
    login,
    logout,
    submit_masked,
    wait_for_round,
)
from scripts.common.secure import seal_round


def read_weights(weights_id: int) -> np.ndarray:
    with Session(engine) as session:
        row = session.get(GlobalWeights, weights_id)
        if row is None:
            raise SystemExit(f"weights {weights_id} not found")
        return np.frombuffer(row.weights, dtype=np.float32)


def run(base: str, key: str, clients: int, rounds: int) -> None:
    with Session(engine) as session:
        version = get_latest_version(session, key)
        if version is None:
            raise SystemExit(f"model '{key}' has no seeded version")
        if version.submission_type is not SubmissionType.secure:
            raise SystemExit(f"model '{key}' is '{version.submission_type.value}', not secure")

    if clients < SECURE_MIN_MEMBERS:
        raise SystemExit(f"--clients {clients} < SECURE_MIN_MEMBERS ({SECURE_MIN_MEMBERS})")

    # Long-term ECDH keypairs, generated once and reused across rounds (the round id
    # in the mask seed keeps each round's masks fresh regardless).
    keypairs = {f"test_{i}": generate_keypair() for i in range(1, clients + 1)}
    rng = np.random.default_rng(SEED)

    print(f"model={key} type=secure clients={clients} rounds={rounds} (no training)")

    for r in range(1, rounds + 1):
        prefix = f"round={r}/{rounds}"

        # Phase A — every client joins and publishes its public key.
        round_id = None
        seats = []  # (user, token, sk, my_user_id)
        for i in range(1, clients + 1):
            user = f"test_{i}"
            token = login(base, user, user)
            sk, pk = keypairs[user]
            resp = join(base, token, key, pk)
            round_id = resp["round_id"]
            seats.append((user, token, sk, resp["user_id"]))

        # Phase B — seal the frozen cohort.
        n = seal_round(round_id, SECURE_MIN_MEMBERS)
        print(f"{prefix} sealed round {round_id} with {n} members")

        # Phase C — random weights, mask, submit. Every client's local weights are
        # W + a random delta bounded by the clip bound (so no clipping distorts the
        # comparison); the delta is what gets quantized and masked.
        desc = get_descriptor(base, seats[0][1], round_id)
        m, B, scale = desc["weight_count"], desc["clip_bound"], desc["scale"]
        base_weights = read_weights(desc["base_weights_id"])
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

        # Client-side proof the masks are antisymmetric: the masked sum equals the
        # unmasked sum exactly (the value the server unmasks to).
        residual = float(np.max(np.abs(
            dequantize(ring_sum(masked_vecs), scale, n)
            - dequantize(ring_sum([quantize(d, B, scale) for d in deltas]), scale, n))))
        print(f"{prefix} mask-cancellation residual: {residual:.3e}")

        # Phase D — aggregate, then read back what the server baked.
        summary = wait_for_round(app.send_task(SECURE_AGG_TASK, args=[round_id]))
        print(f"{prefix} aggregated: {summary}")

        with Session(engine) as session:
            new_weights = np.frombuffer(
                get_latest_weights(session, key).weights, dtype=np.float32)

        # The aggregation result should be W + mean(delta), i.e. the mean of the
        # clients' random weight tensors.
        expected = base_weights + np.mean(np.stack(deltas), axis=0).astype(np.float32)
        max_err = float(np.max(np.abs(new_weights - expected)))
        tol = 2.0 / scale + 1e-4  # quantization error plus the float32 round-trip
        verdict = "OK" if max_err < tol else "MISMATCH"
        print(f"{prefix} aggregate vs plaintext mean: max_err={max_err:.3e} "
              f"tol={tol:.3e} [{verdict}]")
        if max_err >= tol:
            raise SystemExit(f"{prefix} aggregation does not match the plaintext mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', default="cnn-ae", choices=sorted(MODELS), 
                        help="secure-typed model to aggregate for (default: cnn-ae)")
    parser.add_argument("--clients", type=int, default=SECURE_MIN_MEMBERS,
                        help=f"cohort size, one test_N user each (default: {SECURE_MIN_MEMBERS})")
    parser.add_argument("--rounds", type=int, default=1, help="rounds to run (default: 1)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"gateway base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    run(args.base_url, args.model, args.clients, args.rounds)


if __name__ == "__main__":
    main()
