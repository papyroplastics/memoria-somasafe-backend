"""Headless federated client harness — drives the real HTTP API end to end, picking
between two aggregation strategies by the model's ``submission_type``: dense (``raw`` /
``quantize``) has each training subject pull the global weights, train one pass and
upload a plaintext delta for the daily aggregation task; secure runs a round as a
first-class object with join/seal/masked-submit phases and a round-scoped aggregation
task, verifying client-side that the masks cancel. Scores the fresh snapshot on the
held-out subjects each round and writes a convergence CSV + plot."""

import argparse
import base64

import numpy as np
from sqlmodel import Session

from common.celery_tasks import FED_AGG_TASK, SECURE_AGG_TASK
from common.config import DATASETS_DIR, SECURE_MIN_MEMBERS
from common.db import SubmissionType, engine, get_latest_version
from common.secure_agg import (
    dequantize,
    generate_keypair,
    mask_vector,
    quantize,
    ring_sum,
)
from ml.loading import holdout
from ml.model_list import MODELS
from worker.celery_app import app

from scripts.common.api import (
    DEFAULT_BASE_URL,
    download_trainable,
    download_weights,
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
from scripts.common.plots import line_plot
from scripts.common.reports import get_report_dir, write_metrics_csv, write_yaml
from scripts.common.secure import seal_round


class DenseStrategy:
    """raw / quantize: plaintext deltas, averaged by the daily FL task."""
    report_subdir = "fed_client"

    def setup(self, n_clients: int) -> None:
        pass

    def run_round(self, base, key, spec, client, client_datasets, r, rounds, score):
        prefix = f"round={r}/{rounds}"
        scored = False
        for i, dataset in enumerate(client_datasets, start=1):
            user = f"test_{i}"
            token = login(base, user, user)
            raw, weights_id = download_weights(base, token, key)
            client.restore(np.frombuffer(raw, dtype=np.float32))
            base_weights = client.weights()
            if not scored:
                score(client, r - 1)
                scored = True
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
        self.keypairs = {f"test_{i}": generate_keypair() for i in range(1, n_clients + 1)}

    def run_round(self, base, key, spec, client, client_datasets, r, rounds, score):
        prefix = f"round={r}/{rounds}"

        round_id = None
        seats = []
        for i, dataset in enumerate(client_datasets, start=1):
            user = f"test_{i}"
            token = login(base, user, user)
            sk, pk = self.keypairs[user]
            resp = join(base, token, key, pk)
            round_id = resp["round_id"]
            seats.append((i, user, token, dataset, sk, resp["user_id"]))

        n = seal_round(round_id, SECURE_MIN_MEMBERS)
        print(f"{prefix} sealed round {round_id} with {n} members")

        scored = False
        masked_vecs, plain_q, scale = [], [], None
        for i, user, token, dataset, sk, my_id in seats:
            desc = get_descriptor(base, token, round_id)
            raw, weights_id = download_weights(base, token, key)
            if weights_id != desc["base_weights_id"]:
                raise SystemExit("served weights id != round base; client out of sync")
            client.restore(np.frombuffer(raw, dtype=np.float32))
            base_weights = client.weights()
            if not scored:
                score(client, r - 1)
                scored = True
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

        residual = float(np.max(np.abs(
            dequantize(ring_sum(masked_vecs), scale, n)
            - dequantize(ring_sum(plain_q), scale, n))))
        print(f"{prefix} mask-cancellation residual: {residual:.3e}")

        summary = wait_for_round(app.send_task(SECURE_AGG_TASK, args=[round_id]))
        print(f"{prefix} aggregated: {summary}")


def _strategy_for(submission_type: SubmissionType):
    if submission_type is SubmissionType.secure:
        return SecureStrategy()
    return DenseStrategy()


def run(base: str, key: str, rounds: int, eval_subjects: int) -> None:
    spec = MODELS[key]
    strategy = _strategy_for(spec.submission_type)
    trainer = spec.build_trainer(DATASETS_DIR)
    client_datasets, held_out = holdout(trainer.subject_datasets(), eval_subjects)
    eval_data = [dp for ds in held_out for dp in list(ds)]
    strategy.setup(len(client_datasets))

    token = login(base, "test_1", "test_1")
    artifact, _ = download_trainable(base, token, key)
    logout(base, token)
    client = LiteRTClient(artifact, trainer.dataset_tensors)

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
        strategy.run_round(base, key, spec, client, client_datasets, r, rounds, score)

    token = login(base, "test_1", "test_1")
    raw, _ = download_weights(base, token, key)
    logout(base, token)
    client.restore(np.frombuffer(raw, dtype=np.float32))
    score(client, rounds)
    report_dir = get_report_dir(key, strategy.report_subdir)
    metric = trainer.primary_metric
    values = [h[metric] for h in history]
    write_metrics_csv(history, report_dir, "convergence.csv")
    line_plot(report_dir / "convergence.png", [h["round"] for h in history],
              {metric: values}, "round", metric)
    write_yaml(report_dir / "convergence.yaml", {
        'shows': f"Integration-path convergence of {key} ({spec.submission_type.value}), "
                 f"driven over the real HTTP API by the headless "
                 f"{strategy.report_subdir} client.",
        'x_axis': {'name': 'global round', 'range': [0, rounds],
                   'note': '0 = initial global weights, '
                           f'{rounds} = after the last aggregation'},
        'y_axis': {'name': metric, 'better': 'lower' if 'error' in metric else 'higher'},
        'split': {'clients': f'{len(client_datasets)} training subjects (test_N), one '
                             f'submission each per round',
                  'eval_subjects': eval_subjects,
                  'holdout': f'leave-{eval_subjects}-subject-out'},
        'headline': {'start': values[0], 'end': values[-1],
                     'delta': values[-1] - values[0]},
        'purpose': 'integration verification (Sec. 5.1), not a reported convergence curve '
                   '— those come from the simulated federated loop (scripts.system.train '
                   '--loop federated, plotted by scripts.figures.plot_convergence)',
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS), help="model to run the loop for")
    parser.add_argument("--rounds", type=int, default=5, help="global rounds")
    parser.add_argument("--eval-subjects", type=int, default=2,
                        help="subjects reserved from the end for evaluation")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="gateway base URL")
    args = parser.parse_args()

    run(args.base_url, args.model, args.rounds, args.eval_subjects)


if __name__ == "__main__":
    main()
