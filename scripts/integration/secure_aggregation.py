"""Headless secure-aggregation correctness harness — drives the real HTTP API end to end"""

import argparse

import numpy as np
import requests
from sqlmodel import Session

from ml.aggregation import trimmed_mean
from ml.model_list import MODELS
from common.config import (
    FED_TRIM_RATIO,
    SECURE_CLIP_BOUND,
    SECURE_SESSION_MIN_MEMBERS,
    SEED,
)
from common.db import SubmissionType, engine, get_latest_version
from common.ratelimit import clear_model_limits
from common.secure_agg import generate_keypair
from worker.celery_app import app

from scripts.common.api import (
    DEFAULT_BASE_URL,
    download_weights,
    join,
    login,
    logout,
    wait_for_aggregation,
)
from scripts.common.secure import Seat, run_session, split_sessions


def run(base: str, key: str, clients: int, session_size: int, rounds: int) -> None:
    with Session(engine) as session:
        version = get_latest_version(session, key)
        if version is None:
            raise SystemExit(f"model '{key}' has no seeded version")
        if version.submission_type is not SubmissionType.secure:
            raise SystemExit(f"model '{key}' is '{version.submission_type.value}', not secure")

    users = [f"test_{i}" for i in range(1, clients + 1)]
    keypairs = {user: generate_keypair() for user in users}
    rng = np.random.default_rng(SEED)

    print(f"model={key} type=secure clients={clients} session_size={session_size} "
          f"rounds={rounds} (no training)")

    for r in range(1, rounds + 1):
        prefix = f"round={r}/{rounds}"
        clear_model_limits(key)
        tokens = {user: login(base, user, user) for user in users}
        raw, weights_id = download_weights(base, tokens[users[0]], key)
        base_weights = np.frombuffer(raw, dtype=np.float32)
        seats = [Seat(user, tokens[user], *keypairs[user],
                      rng.uniform(-SECURE_CLIP_BOUND, SECURE_CLIP_BOUND,
                                  base_weights.size).astype(np.float32))
                 for user in users]

        results = []
        for group in split_sessions(seats, session_size, SECURE_SESSION_MIN_MEMBERS):
            result = run_session(app, base, key, weights_id, group, SECURE_SESSION_MIN_MEMBERS)
            print(f"{prefix} session {result.session_id}: {result.members} members, "
                  f"mask-cancellation residual {result.residual:.3e}")
            results.append(result)

        clear_model_limits(key)
        try:
            join(base, tokens[users[0]], key, weights_id, keypairs[users[0]][1])
            raise SystemExit(f"{prefix} a second join on the same weights was accepted")
        except requests.HTTPError as exc:
            if exc.response.status_code != 409:
                raise
        print(f"{prefix} second join on the same weights rejected (409)")

        summary = wait_for_aggregation(app, key)
        print(f"{prefix} aggregated: {summary['cohort']} sessions, "
              f"{summary['submissions']} submissions")

        clear_model_limits(key)
        raw, _ = download_weights(base, tokens[users[0]], key)
        for token in tokens.values():
            logout(base, token)
        new_weights = np.frombuffer(raw, dtype=np.float32)

        expected = base_weights + trimmed_mean([res.mean for res in results], FED_TRIM_RATIO)
        max_err = float(np.max(np.abs(new_weights - expected)))
        tol = 2.0 / min(res.scale for res in results) + 1e-4
        verdict = "OK" if max_err < tol else "MISMATCH"
        print(f"{prefix} aggregate vs plaintext trimmed mean: max_err={max_err:.3e} "
              f"tol={tol:.3e} [{verdict}]")
        if max_err >= tol:
            raise SystemExit(f"{prefix} aggregation does not match the plaintext trimmed mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', nargs='?', default="feature-ae-secure",
                        choices=sorted(MODELS),
                        help="secure-typed model to aggregate for")
    parser.add_argument("--clients", type=int, default=9,
                        help="clients per round, one test_N user each")
    parser.add_argument("--session-size", type=int, default=SECURE_SESSION_MIN_MEMBERS,
                        help="target members per session")
    parser.add_argument("--rounds", type=int, default=1, help="rounds to run")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="gateway base URL")
    args = parser.parse_args()

    run(args.base_url, args.model, args.clients, args.session_size, args.rounds)


if __name__ == "__main__":
    main()
