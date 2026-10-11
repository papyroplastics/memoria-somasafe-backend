"""Round lifecycle harness: full rounds close and aggregate on their own, without waiting for
sessions still in flight, and the cleanup sweep leaves no submitted vector behind."""

import argparse
import time
from contextlib import contextmanager

import numpy as np
import requests
from sqlalchemy import func
from sqlmodel import Session, select

from common.compression import decompress
from common.config import FED_TRIM_RATIO, SECURE_CLIP_BOUND, SECURE_SESSION_MIN_MEMBERS, SEED
from common.db import (
    ClientDeltaSubmission,
    ModelVersion,
    SecurePartial,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    SubmissionType,
    count_summed_members,
    engine,
    get_latest_version,
    get_latest_weights,
    utcnow,
)
from common.ratelimit import clear_model_limits
from common.secure_agg import generate_keypair
from ml.aggregation import trimmed_mean
from worker.celery_app import app
from worker.compute import Action, SweepAction
from worker.tasks.cleanup import cleanup_results
from worker.tasks.secure import _fail

from scripts.common.api import (DEFAULT_BASE_URL, download_weights, join, login, logout,
                                submit_delta, submit_masked)
from scripts.common.secure import Seat, join_session, run_session

AGGREGATION_TIMEOUT = 120.0


def _require(key: str, types: set[SubmissionType]) -> None:
    with Session(engine) as session:
        version = get_latest_version(session, key)
    if version is None:
        raise SystemExit(f"model '{key}' has no seeded version")
    if version.submission_type not in types:
        raise SystemExit(f"model '{key}' is '{version.submission_type.value}'")


@contextmanager
def round_limit(key: str, limit: int):
    with Session(engine) as session:
        version = get_latest_version(session, key)
        original, version.round_limit = version.round_limit, limit
        session.add(version)
        session.commit()
        version_id = version.id
    try:
        yield
    finally:
        with Session(engine) as session:
            version = session.get(ModelVersion, version_id)
            version.round_limit = original
            session.add(version)
            session.commit()


def _weights(key: str) -> tuple[int, np.ndarray]:
    with Session(engine) as session:
        weights = get_latest_weights(session, key)
        return weights.id, np.frombuffer(decompress(weights.weights), dtype=np.float32)


def _wait_for_round(key: str, base_id: int) -> np.ndarray:
    deadline = time.monotonic() + AGGREGATION_TIMEOUT
    while time.monotonic() < deadline:
        weights_id, weights = _weights(key)
        if weights_id != base_id:
            return weights
        time.sleep(0.5)
    raise SystemExit(f"{key}: the full round was not aggregated within {AGGREGATION_TIMEOUT:.0f}s")


def _check(key: str, actual: np.ndarray, expected: np.ndarray, tol: float) -> None:
    max_err = float(np.max(np.abs(actual - expected)))
    verdict = "OK" if max_err < tol else "MISMATCH"
    print(f"{key}: aggregate vs plaintext trimmed mean: max_err={max_err:.3e} tol={tol:.3e} [{verdict}]")
    if max_err >= tol:
        raise SystemExit(f"{key}: aggregation does not match the plaintext trimmed mean")


def _left(model, *where) -> int:
    with Session(engine) as session:
        return session.exec(select(func.count()).select_from(model).where(*where)).one()


def run_dense(base: str, key: str, clients: int) -> None:
    _require(key, {SubmissionType.raw, SubmissionType.quantize})
    clear_model_limits(key)
    users = [f"test_{i}" for i in range(1, clients + 1)]
    tokens = {user: login(base, user, user) for user in users}
    raw, base_id = download_weights(base, tokens[users[0]], key)
    reference = np.frombuffer(raw, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    deltas = [(rng.standard_normal(reference.size) * 1e-3).astype(np.float32) for _ in users]

    existing = _left(ClientDeltaSubmission, ClientDeltaSubmission.base_weights_id == base_id)
    with round_limit(key, existing + clients):
        for user, delta in zip(users, deltas):
            submit_delta(base, tokens[user], key, base_id, delta.tobytes(), SubmissionType.raw)
        print(f"{key}: {clients} submissions on weights {base_id}, waiting for the round")
        weights = _wait_for_round(key, base_id)
    _check(key, weights, reference + trimmed_mean(deltas, FED_TRIM_RATIO), 1e-6)

    try:
        submit_delta(base, tokens[users[0]], key, base_id, deltas[0].tobytes(), SubmissionType.raw)
        raise SystemExit(f"{key}: a submission on the aggregated weights was accepted")
    except requests.HTTPError as exc:
        if exc.response.status_code != 409:
            raise
    for token in tokens.values():
        logout(base, token)

    cleanup_results()
    left = _left(ClientDeltaSubmission, ClientDeltaSubmission.base_weights_id == base_id,
                 ClientDeltaSubmission.deltas.is_not(None))  # type: ignore[union-attr]
    rows = _left(ClientDeltaSubmission, ClientDeltaSubmission.base_weights_id == base_id)
    print(f"{key}: after cleanup, {left} of {rows} submissions still hold deltas")
    if left or rows < existing + clients:
        raise SystemExit(f"{key}: expected {clients} kept submissions with cleared deltas")


def _fail_open_sessions(key: str, base_id: int) -> None:
    with Session(engine) as session:
        stray = session.exec(select(SecureSession.id).where(
            SecureSession.base_weights_id == base_id,
            SecureSession.status == SecureSessionStatus.open)).all()
        for session_id in stray:
            _fail(session, SweepAction(session_id, key, Action.fail, SecureSessionStatus.open), utcnow())
        session.commit()
    if stray:
        print(f"{key}: failed {len(stray)} open session(s) left on weights {base_id}")


def _rejected(call) -> bool:
    try:
        call()
        return False
    except requests.HTTPError as exc:
        if exc.response.status_code != 409:
            raise
        return True


def run_secure(base: str, key: str) -> None:
    _require(key, {SubmissionType.secure})
    clear_model_limits(key)
    size = SECURE_SESSION_MIN_MEMBERS
    users = [f"test_{i}" for i in range(1, 2 * size + 2)]
    tokens = {user: login(base, user, user) for user in users}
    raw, base_id = download_weights(base, tokens[users[0]], key)
    reference = np.frombuffer(raw, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    seats = [Seat(user, tokens[user], *generate_keypair(),
                  rng.uniform(-SECURE_CLIP_BOUND, SECURE_CLIP_BOUND, reference.size).astype(np.float32))
             for user in users]
    closing, in_flight, latecomer = seats[:size], seats[size:2 * size], seats[-1]

    _fail_open_sessions(key, base_id)
    with Session(engine) as session:
        summed = count_summed_members(session, base_id)
    with round_limit(key, summed + size):
        pending_id, _, _ = join_session(base, key, base_id, in_flight, size)
        print(f"{key}: session {pending_id} sealed and left waiting for submissions")
        result = run_session(app, base, key, base_id, closing, size)
        print(f"{key}: session {result.session_id} summed ({result.members} members), round full")

        if not _rejected(lambda: join(base, latecomer.token, key, base_id, latecomer.pk)):
            raise SystemExit(f"{key}: a join on the full round was accepted")
        stuck = in_flight[0]
        body = np.zeros(reference.size, dtype="<u4").tobytes()
        if not _rejected(lambda: submit_masked(base, stuck.token, pending_id, body)):
            raise SystemExit(f"{key}: a submission to a session of the closed round was accepted")
        print(f"{key}: late join and in-flight submission rejected (409)")
        weights = _wait_for_round(key, base_id)
    tol = 2.0 / result.scale + 1e-4
    _check(key, weights, reference + trimmed_mean([result.mean], FED_TRIM_RATIO), tol)
    for token in tokens.values():
        logout(base, token)

    cleanup_results()
    on_base = select(SecureSession.id).where(SecureSession.base_weights_id == base_id)
    partials = _left(SecurePartial, SecurePartial.session_id.in_(on_base))  # type: ignore[attr-defined]
    masked = _left(SecureSessionMember, SecureSessionMember.base_weights_id == base_id,
                   SecureSessionMember.masked.is_not(None))  # type: ignore[union-attr]
    print(f"{key}: after cleanup, {partials} partials and {masked} masked vectors left")
    if partials or masked:
        raise SystemExit(f"{key}: the finished round still holds submitted vectors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dense", default="feature-ae", help="raw or quantize model")
    parser.add_argument("--secure", default="feature-ae-secure", help="secure model")
    parser.add_argument("--clients", type=int, default=5, help="dense round limit")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="gateway base URL")
    args = parser.parse_args()

    run_dense(args.base_url, args.dense, args.clients)
    run_secure(args.base_url, args.secure)
    print("OK")


if __name__ == "__main__":
    main()
