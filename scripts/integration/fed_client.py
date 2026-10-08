"""Headless federated client harness — drives the real HTTP API end to end"""

import argparse

import numpy as np
from sqlmodel import Session

from common.config import DATASETS_DIR, SECURE_SESSION_MAX_MEMBERS, SECURE_SESSION_MIN_MEMBERS
from common.db import SubmissionType, engine, get_latest_version
from common.secure_agg import generate_keypair
from ml.sources.common import holdout
from ml.model_list import MODELS
from worker.celery_app import app

from scripts.common.api import (
    DEFAULT_BASE_URL,
    download_trainable,
    download_weights,
    login,
    logout,
    submit_delta,
    wait_for_aggregation,
)
from scripts.common.litert import LiteRTClient
from scripts.common.plots import line_plot
from scripts.common.reports import get_report_dir, write_metrics_csv, write_yaml
from scripts.common.secure import Seat, run_session, split_sessions


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
        result = wait_for_aggregation(app, key)
        print(f"{prefix} aggregated: cohort {result['cohort']}, cap {result['cap']}")


class SecureStrategy:
    """secure: every client trains first, then joins a session on the weights it
    trained on; each session's masked sum becomes a partial, and the round task
    trimmed-means the partials."""

    report_subdir = "secure_fed_client"

    def __init__(self, session_size: int) -> None:
        self.session_size = session_size

    def setup(self, n_clients: int) -> None:
        if n_clients < SECURE_SESSION_MIN_MEMBERS:
            raise SystemExit(f"{n_clients} client subjects < SECURE_SESSION_MIN_MEMBERS "
                             f"({SECURE_SESSION_MIN_MEMBERS}); a session needs at least that many")
        self.keypairs = {f"test_{i}": generate_keypair() for i in range(1, n_clients + 1)}

    def run_round(self, base, key, spec, client, client_datasets, r, rounds, score):
        prefix = f"round={r}/{rounds}"

        scored = False
        seats, weights_id = [], None
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
            delta = (client.weights() - base_weights).astype(np.float32)
            seats.append(Seat(user, token, *self.keypairs[user], delta))

        for group in split_sessions(seats, self.session_size, SECURE_SESSION_MIN_MEMBERS):
            result = run_session(app, base, key, weights_id, group, SECURE_SESSION_MIN_MEMBERS)
            print(f"{prefix} session {result.session_id}: {result.members} members, "
                  f"mask-cancellation residual {result.residual:.3e}")
        for seat in seats:
            logout(base, seat.token)

        summary = wait_for_aggregation(app, key)
        print(f"{prefix} aggregated: {summary['cohort']} sessions, "
              f"{summary['submissions']} submissions")


def _strategy_for(submission_type: SubmissionType, session_size: int):
    if submission_type is SubmissionType.secure:
        return SecureStrategy(session_size)
    return DenseStrategy()


def run(base: str, key: str, rounds: int, eval_subjects: int, session_size: int) -> None:
    spec = MODELS[key]
    strategy = _strategy_for(spec.submission_type, session_size)
    trainer = spec.trainer_cls(spec.model_cls(), DATASETS_DIR)
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
        'model': key,
        'submission_type': spec.submission_type.value,
        'metric': metric,
        'clients': len(client_datasets),
        'eval_subjects': eval_subjects,
        'rounds': rounds,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS), help="model to run the loop for")
    parser.add_argument("--rounds", type=int, default=5, help="global rounds")
    parser.add_argument("--eval-subjects", type=int, default=2,
                        help="subjects reserved from the end for evaluation")
    parser.add_argument("--session-size", type=int, default=SECURE_SESSION_MAX_MEMBERS,
                        help="target members per secure session (secure models only)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="gateway base URL")
    args = parser.parse_args()

    run(args.base_url, args.model, args.rounds, args.eval_subjects, args.session_size)


if __name__ == "__main__":
    main()
