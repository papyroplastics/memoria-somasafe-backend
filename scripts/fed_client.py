"""Headless federated client harness — drives the real HTTP API end to end.

One harness, two aggregation strategies picked by the model's ``submission_type``:

  - **dense** (``raw`` / ``quantize``): each round every training subject (as user
    ``test_N``) logs in, pulls the current global trainable artifact, trains one pass
    through the on-device LiteRT ``CompiledModel`` runtime, uploads the plaintext
    weight delta (local − global), and logs out. One ``federated_aggregation`` task
    then averages whatever landed since the last snapshot.
  - **secure**: a secure round is a first-class object, so the round runs the four
    synchronised phases the masking protocol needs — join (publish an ECDH key and
    take a seat), seal (freeze the cohort + scale), masked submit (each member masks
    its quantized delta against the frozen roster and uploads only the masked
    vector), then one round-scoped ``secure_aggregation`` task sums them. The harness
    also verifies client-side that the masks cancel exactly each round.

Either way it scores the fresh snapshot on the held-out subjects and writes a
per-round convergence CSV + plot.

    uv run -m scripts.fed_client --model feature-mlp --rounds 5 --eval-subjects 2
    uv run -m scripts.fed_client --model cnn-ae     --rounds 5 --eval-subjects 2   # secure

Prereqs: services up (api, worker, redis, postgres), DB seeded with ``--test-users``,
and the model trained/seeded. There are no local epochs — one submission per client
per round.
"""

import argparse
import base64

import numpy as np
from sqlmodel import Session

from common.config import DATASETS_DIR, MODELS_DIR, SECURE_MIN_MEMBERS
from common.db import SubmissionType, engine, get_latest_version
from common.secure_agg import (
    dequantize,
    generate_keypair,
    mask_vector,
    quantize,
    ring_sum,
)
from ml.model_list import MODELS
from worker.celery_app import app

from scripts.common.api import (
    DEFAULT_BASE_URL,
    download_trainable,
    get_descriptor,
    join,
    login,
    logout,
    submit_delta,
    submit_masked,
    wait_for_aggregation,
    wait_for_round,
)
from scripts.common.litert import LiteRTClient
from scripts.common.post_train import get_report_dir, plot_metric, write_metrics_csv
from scripts.common.secure import seal_round

FED_AGG_TASK = "worker.tasks.federated_aggregation"
SECURE_AGG_TASK = "worker.tasks.secure_aggregation"


class DenseStrategy:
    """raw / quantize: plaintext deltas, averaged by the daily FedAvg task."""

    report_subdir = "fed_client"

    def setup(self, n_clients: int) -> None:
        pass

    def run_round(self, base, key, spec, trainer, client_datasets, r, rounds, score):
        prefix = f"round={r}/{rounds}"
        round_global = None
        for i, dataset in enumerate(client_datasets, start=1):
            user = f"test_{i}"
            token = login(base, user, user)
            artifact, weights_id = download_trainable(base, token, key)
            client = LiteRTClient(artifact, trainer.dataset_tensors)
            base_weights = client.weights()  # global snapshot the delta is relative to
            if round_global is None:
                round_global = artifact
                score(client, r - 1)  # global weights this round trained from
            client.train_pass(dataset, f"{prefix} subject={i}/{len(client_datasets)}")
            delta = client.weights() - base_weights
            submit_delta(base, token, key, weights_id,
                         delta.astype(np.float32).tobytes(), spec.submission_type)
            logout(base, token)
        summary = wait_for_aggregation(app.send_task(FED_AGG_TASK, args=[key]), key)
        print(f"{prefix} aggregated: {summary}")


class SecureStrategy:
    """secure: a first-class round — join, freeze the cohort, then every member
    uploads a masked delta and the round-scoped task sums them (masks cancel)."""

    report_subdir = "secure_fed_client"

    def setup(self, n_clients: int) -> None:
        if n_clients < SECURE_MIN_MEMBERS:
            raise SystemExit(f"{n_clients} client subjects < SECURE_MIN_MEMBERS "
                             f"({SECURE_MIN_MEMBERS}); a secure round needs at least that many")
        # Long-term ECDH keypairs, generated once and reused across rounds (the round
        # id in the mask seed keeps each round's masks fresh regardless).
        self.keypairs = {f"test_{i}": generate_keypair() for i in range(1, n_clients + 1)}

    def run_round(self, base, key, spec, trainer, client_datasets, r, rounds, score):
        prefix = f"round={r}/{rounds}"

        # Phase A — every client joins and publishes its public key.
        round_id = None
        seats = []  # (i, user, token, dataset, sk, my_user_id)
        for i, dataset in enumerate(client_datasets, start=1):
            user = f"test_{i}"
            token = login(base, user, user)
            sk, pk = self.keypairs[user]
            resp = join(base, token, key, pk)
            round_id = resp["round_id"]
            seats.append((i, user, token, dataset, sk, resp["user_id"]))

        # Phase B — seal the frozen cohort.
        n = seal_round(round_id, SECURE_MIN_MEMBERS)
        print(f"{prefix} sealed round {round_id} with {n} members")

        # Phase C — train, mask, submit. Collected q/y let us confirm the masks
        # cancel to the same sum the server will compute.
        round_global = None
        masked_vecs, plain_q, scale = [], [], None
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
            client.train_pass(dataset, f"{prefix} subject={i}/{len(seats)}")
            delta = client.weights() - base_weights
            scale = desc["scale"]
            q = quantize(delta, desc["clip_bound"], scale)
            roster = [(e["user_id"], base64.b64decode(e["ka_public_key"]))
                      for e in desc["roster"]]
            y = mask_vector(q, my_id, roster, sk, round_id)
            submit_masked(base, token, round_id, y.astype("<u4").tobytes())
            logout(base, token)
            masked_vecs.append(y)
            plain_q.append(q)

        # Client-side proof the masks are antisymmetric: the masked sum equals the
        # unmasked sum exactly (the value the server unmasks to).
        residual = float(np.max(np.abs(
            dequantize(ring_sum(masked_vecs), scale, n)
            - dequantize(ring_sum(plain_q), scale, n))))
        print(f"{prefix} mask-cancellation residual: {residual:.3e}")

        # Phase D — aggregate.
        summary = wait_for_round(app.send_task(SECURE_AGG_TASK, args=[round_id]))
        print(f"{prefix} aggregated: {summary}")


def _strategy_for(submission_type: SubmissionType):
    if submission_type is SubmissionType.secure:
        return SecureStrategy()
    return DenseStrategy()


def _split_eval(subject_datasets, eval_subjects):
    if eval_subjects >= len(subject_datasets):
        raise SystemExit(f"--eval-subjects {eval_subjects} leaves no training subjects "
                         f"({len(subject_datasets)} available)")
    if eval_subjects <= 0:
        return subject_datasets, None
    # Materialize each held-out subject and concatenate at the Python level, so the
    # already-cached subject datasets are never re-combined into a new tf pipeline.
    eval_data = [dp for ds in subject_datasets[-eval_subjects:] for dp in list(ds)]
    return subject_datasets[:-eval_subjects], eval_data


def run(base: str, key: str, rounds: int, eval_subjects: int) -> None:
    spec = MODELS[key]
    strategy = _strategy_for(spec.submission_type)
    trainer = spec.build_trainer(DATASETS_DIR)
    subject_datasets, _ = trainer.subject_datasets(DATASETS_DIR, train_split=1.0)
    client_datasets, eval_data = _split_eval(subject_datasets, eval_subjects)
    strategy.setup(len(client_datasets))

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
        with Session(engine) as session:
            if get_latest_version(session, key) is None:
                raise SystemExit(f"model '{key}' has no seeded version")
        strategy.run_round(base, key, spec, trainer, client_datasets, r, rounds, score)

    # Final global weights, after the last round's aggregation.
    token = login(base, "test_1", "test_1")
    artifact, _ = download_trainable(base, token, key)
    logout(base, token)
    score(LiteRTClient(artifact, trainer.dataset_tensors), rounds)

    report_dir = get_report_dir(MODELS_DIR / key, strategy.report_subdir)
    write_metrics_csv(history, report_dir, "convergence.csv")
    plot_metric(history, "round", trainer.primary_metric, report_dir, "convergence.png")


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
    args = parser.parse_args()

    run(args.base_url, args.model, args.rounds, args.eval_subjects)


if __name__ == "__main__":
    main()
