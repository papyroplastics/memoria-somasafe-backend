"""Secure-session dropout harness: one member of a sealed session never submits, the
sweep fails the session once its seal timeout has passed (backdated here), and the
members who did submit can rejoin at once while the dropout stays on cooldown."""

import argparse
from datetime import timedelta

import numpy as np
import requests
from sqlmodel import Session, select

from common.config import SECURE_SESSION_MIN_MEMBERS, SECURE_SESSION_SEALED_FAIL_TIMEOUT_SECONDS
from common.db import (
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    SubmissionType,
    engine,
    get_latest_version,
    utcnow,
)
from common.ratelimit import clear_model_limits
from common.secure_agg import generate_keypair
from ml.model_list import MODELS
from worker.tasks.secure import secure_session_sweep

from scripts.common.api import (
    DEFAULT_BASE_URL,
    download_weights,
    join,
    login,
    logout,
    submit_masked,
)
from scripts.common.secure import seal_session


def _join_status(base: str, token: str, key: str, weights_id: int) -> int:
    try:
        join(base, token, key, weights_id, generate_keypair()[1])
        return 202
    except requests.HTTPError as exc:
        return exc.response.status_code


def run(base: str, key: str) -> None:
    with Session(engine) as session:
        version = get_latest_version(session, key)
        if version is None:
            raise SystemExit(f"model '{key}' has no seeded version")
        if version.submission_type is not SubmissionType.secure:
            raise SystemExit(f"model '{key}' is '{version.submission_type.value}', not secure")

    clear_model_limits(key)
    users = [f"test_{i}" for i in range(1, SECURE_SESSION_MIN_MEMBERS + 1)]
    tokens = {user: login(base, user, user) for user in users}
    raw, weights_id = download_weights(base, tokens[users[0]], key)
    weight_count = np.frombuffer(raw, dtype=np.float32).size

    session_id = None
    for user in users:
        session_id = join(base, tokens[user], key, weights_id,
                          generate_keypair()[1])["session_id"]
    n = seal_session(session_id, SECURE_SESSION_MIN_MEMBERS)
    survivors, dropout = users[:-1], users[-1]
    for user in survivors:
        submit_masked(base, tokens[user], session_id,
                      np.zeros(weight_count, dtype="<u4").tobytes())
    print(f"session {session_id}: {n} members, {len(survivors)} submitted, {dropout} dropped")

    with Session(engine) as session:
        row = session.get(SecureSession, session_id)
        row.sealed_at = utcnow() - timedelta(seconds=SECURE_SESSION_SEALED_FAIL_TIMEOUT_SECONDS + 60)
        session.add(row)
        session.commit()
    secure_session_sweep()

    with Session(engine) as session:
        status = session.get(SecureSession, session_id).status
        seats = session.exec(select(SecureSessionMember)
                             .where(SecureSessionMember.session_id == session_id)).all()
    print(f"after sweep: status={status.value}, seats left={len(seats)}")
    if status is not SecureSessionStatus.failed or seats:
        raise SystemExit("the sweep did not fail the session and release its seats")

    survivor_status = _join_status(base, tokens[survivors[0]], key, weights_id)
    dropout_status = _join_status(base, tokens[dropout], key, weights_id)
    print(f"survivor rejoin: {survivor_status}, dropout rejoin: {dropout_status}")
    for token in tokens.values():
        logout(base, token)
    if survivor_status != 202 or dropout_status != 429:
        raise SystemExit("expected the survivor to rejoin (202) and the dropout to be on "
                         "cooldown (429)")
    print("OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', nargs='?', default="feature-ae-secure",
                        choices=sorted(MODELS),
                        help="secure-typed model to run the dropout for")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="gateway base URL")
    args = parser.parse_args()

    run(args.base_url, args.model)


if __name__ == "__main__":
    main()
