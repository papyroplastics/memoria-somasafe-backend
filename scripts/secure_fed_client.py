"""Headless secure-aggregation client harness — drives the real HTTP API end to
end, the masked counterpart of ``scripts.fed_client``.

Each round runs the three synchronised phases the protocol needs (unlike the
async submit path, a secure round is a first-class object with a frozen cohort):

  A. join   — every subject (as user ``test_N``) logs in, publishes its long-term
              ECDH public key, and takes a seat in the round.
  B. seal   — the roster is frozen and the fixed-point scale S = floor(2^31/(n*B))
              is set (done here directly against the DB; there is no seal endpoint,
              so a client can never seal a roster mid-join). Needs n >= SECURE_MIN_MEMBERS.
  C. mask   — every member pulls the round descriptor and the base weights W, trains
              one pass, quantizes and masks its delta against the frozen roster, and
              uploads only the masked vector.
  D. aggregate — one ``secure_aggregation`` task sums the masked vectors (the masks
              cancel), dequantizes, and bakes the new global weights.

Between the masked submit and the round result nothing individual is ever exposed;
the harness additionally verifies client-side that the masks cancel exactly (the
same sum the server computes) each round.

    uv run -m scripts.secure_fed_client --rounds 5 --eval-subjects 2

Prereqs: services up (api, worker, redis, postgres), DB seeded with ``--test-users``,
and the model trained/seeded with ``submission_type = secure`` (cnn-ae by default).
"""

import argparse
import base64

import numpy as np
import requests
from sqlmodel import Session, select

from common.config import DATASETS_DIR, MODELS_DIR, SECURE_MIN_MEMBERS
from common.db import (
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    engine,
    get_latest_version,
    utcnow,
)
from scripts.common.post_train import get_report_dir, plot_metric, write_metrics_csv
from common.secure_agg import (
    compute_scale,
    dequantize,
    generate_keypair,
    mask_vector,
    quantize,
    ring_sum,
)
from ml.model_list import MODELS
from worker.celery_app import app

# Reuse the LiteRT training/eval driver and the auth/download helpers verbatim.
from scripts.fed_client import (
    LiteRTClient,
    _auth,
    download_trainable,
    login,
    logout,
)

SECURE_AGG_TASK = "worker.tasks.secure_aggregation"
DEFAULT_BASE_URL = "http://localhost:8000"


def join(base: str, token: str, key: str, ka_public_key: bytes) -> dict:
    resp = requests.post(
        f"{base}/model/secure/join/{key}", headers=_auth(token),
        json={"ka_public_key": base64.b64encode(ka_public_key).decode()},
    )
    resp.raise_for_status()
    return resp.json()


def get_descriptor(base: str, token: str, round_id: int) -> dict:
    resp = requests.get(f"{base}/model/secure/round/{round_id}", headers=_auth(token))
    resp.raise_for_status()
    return resp.json()


def submit_masked(base: str, token: str, round_id: int, body: bytes) -> None:
    resp = requests.post(
        f"{base}/model/secure/submit/{round_id}",
        headers=_auth(token) | {"Content-Type": "application/octet-stream"},
        data=body,
    )
    resp.raise_for_status()


def seal_round(round_id: int, min_members: int) -> int:
    """Freeze the roster and fix n and the scale S. Done directly against the DB —
    the harness plays the operator that a real deployment would automate."""
    with Session(engine) as session:
        round = session.get(SecureRound, round_id)
        if round is None:
            raise SystemExit(f"round {round_id} vanished before seal")
        members = list(session.exec(
            select(SecureRoundMember).where(SecureRoundMember.round_id == round_id)))
        n = len(members)
        if n < min_members:
            raise SystemExit(f"only {n} members joined, need >= {min_members} to seal")
        round.member_count = n
        round.scale = compute_scale(n, round.clip_bound)
        round.status = SecureRoundStatus.sealed
        round.sealed_at = utcnow()
        session.add(round)
        session.commit()
        return n


def wait_for_round(result, timeout: float = 300.0) -> str:
    summary = result.get(timeout=timeout)
    if summary.startswith(("skipped", "failed")) or "export failed" in summary:
        raise SystemExit(f"secure round produced no new weights: {summary}")
    return summary


def run(base: str, key: str, rounds: int, eval_subjects: int) -> None:
    spec = MODELS[key]
    if spec.submission_type.value != "secure":
        raise SystemExit(f"model '{key}' is '{spec.submission_type.value}', not secure")

    trainer = spec.build_trainer(DATASETS_DIR)
    subject_datasets, _ = trainer.subject_datasets(DATASETS_DIR, train_split=1.0)

    if eval_subjects >= len(subject_datasets):
        raise SystemExit(f"--eval-subjects {eval_subjects} leaves no training subjects "
                         f"({len(subject_datasets)} available)")
    if eval_subjects > 0:
        client_datasets = subject_datasets[:-eval_subjects]
        eval_data = [dp for ds in subject_datasets[-eval_subjects:] for dp in list(ds)]
    else:
        client_datasets = subject_datasets
        eval_data = None

    if len(client_datasets) < SECURE_MIN_MEMBERS:
        raise SystemExit(f"{len(client_datasets)} client subjects < SECURE_MIN_MEMBERS "
                         f"({SECURE_MIN_MEMBERS}); a secure round needs at least that many")

    # Long-term ECDH keypairs, generated once and reused across rounds (the round id
    # in the mask seed keeps each round's masks fresh regardless).
    keypairs = {f"test_{i}": generate_keypair()
                for i in range(1, len(client_datasets) + 1)}

    print(f"model={key} type=secure clients={len(client_datasets)} "
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
        round_prefix = f"round={r}/{rounds}"
        with Session(engine) as session:
            if get_latest_version(session, key) is None:
                raise SystemExit(f"model '{key}' has no seeded version")

        # Phase A — every client joins and publishes its public key.
        round_id = None
        seats = []  # (idx, user, token, dataset, sk, my_user_id)
        for i, dataset in enumerate(client_datasets, start=1):
            user = f"test_{i}"
            token = login(base, user, user)
            sk, pk = keypairs[user]
            resp = join(base, token, key, pk)
            round_id = resp["round_id"]
            seats.append((i, user, token, dataset, sk, resp["user_id"]))

        # Phase B — seal the frozen cohort.
        n = seal_round(round_id, SECURE_MIN_MEMBERS)
        print(f"{round_prefix} sealed round {round_id} with {n} members")

        # Phase C — train, mask, submit. Collected q/y let us confirm the masks
        # cancel to the same sum the server will compute.
        round_global = None
        masked_vecs, plain_q = [], []
        for i, user, token, dataset, sk, my_id in seats:
            desc = get_descriptor(base, token, round_id)
            artifact, weights_id = download_trainable(base, token, key)
            if weights_id != desc["base_weights_id"]:
                raise SystemExit("served weights id != round base; client out of sync")

            client = LiteRTClient(artifact, trainer.dataset_tensors)
            base_weights = client.weights()
            if round_global is None:
                round_global = artifact
                score(client, r - 1)

            client.train_pass(dataset, f"{round_prefix} subject={i}/{len(seats)}")
            delta = client.weights() - base_weights

            q = quantize(delta, desc["clip_bound"], desc["scale"])
            roster = [(e["user_id"], base64.b64decode(e["ka_public_key"]))
                      for e in desc["roster"]]
            y = mask_vector(q, my_id, roster, sk, round_id)
            submit_masked(base, token, round_id, y.astype("<u4").tobytes())
            logout(base, token)
            masked_vecs.append(y)
            plain_q.append(q)

        # Client-side proof the masks are antisymmetric: the masked sum equals the
        # unmasked sum exactly (this is the value the server unmasks to).
        residual = float(np.max(np.abs(
            dequantize(ring_sum(masked_vecs), desc["scale"], n)
            - dequantize(ring_sum(plain_q), desc["scale"], n))))
        print(f"{round_prefix} mask-cancellation residual: {residual:.3e}")

        # Phase D — aggregate.
        summary = wait_for_round(app.send_task(SECURE_AGG_TASK, args=[round_id]))
        print(f"round={r} aggregated: {summary}")

    # Final global weights, after the last round.
    token = login(base, "test_1", "test_1")
    artifact, _ = download_trainable(base, token, key)
    logout(base, token)
    score(LiteRTClient(artifact, trainer.dataset_tensors), rounds)

    report_dir = get_report_dir(MODELS_DIR / key, "secure_fed_client")
    write_metrics_csv(history, report_dir, "convergence.csv")
    plot_metric(history, "round", trainer.primary_metric, report_dir, "convergence.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="cnn-ae", choices=sorted(MODELS),
                        help="secure-typed model to run the loop for (default: cnn-ae)")
    parser.add_argument("--rounds", type=int, default=5, help="global rounds (default: 5)")
    parser.add_argument("--eval-subjects", type=int, default=2,
                        help="subjects reserved from the end for evaluation (default: 2)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"gateway base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    run(args.base_url, args.model, args.rounds, args.eval_subjects)


if __name__ == "__main__":
    main()
