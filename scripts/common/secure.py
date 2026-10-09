"""Secure-session helpers for the harnesses. Sessions are sealed and summed by hand
for determinism, tolerating the worker's sweep having got there first."""

import base64
import time
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec
from sqlmodel import Session

from common.celery_tasks import SECURE_SUM_TASK
from common.db import SecureSession, SecureSessionStatus, engine
from common.db import seal_session as seal
from common.secure_agg import dequantize, mask_vector, quantize, ring_sum

from scripts.common.api import get_descriptor, join, submit_masked


@dataclass(frozen=True)
class Seat:
    user: str
    token: str
    sk: ec.EllipticCurvePrivateKey
    pk: bytes
    delta: np.ndarray


@dataclass(frozen=True)
class SessionResult:
    session_id: int
    members: int
    scale: int
    mean: np.ndarray
    residual: float


def split_sessions(seats: list[Seat], size: int, min_members: int) -> list[list[Seat]]:
    if len(seats) < min_members:
        raise SystemExit(f"{len(seats)} clients < {min_members}, the minimum session size")
    count = min(-(-len(seats) // size), len(seats) // min_members)
    return [list(group) for group in np.array_split(np.array(seats, dtype=object), count)]


def seal_session(session_id: int, min_members: int) -> int:
    with Session(engine) as session:
        sealed = seal(session, session_id)
        if sealed is not None and sealed < min_members:
            raise SystemExit(f"only {sealed} members joined, need >= {min_members} to seal")
        session.commit()
        if sealed is not None:
            return sealed
        current = session.get(SecureSession, session_id, populate_existing=True)
        if current is None or current.status is not SecureSessionStatus.sealed:
            raise SystemExit(f"session {session_id} could not be sealed "
                             f"({current.status.value if current else 'missing'})")
        return current.member_count


def sum_session(app, session_id: int, timeout: float = 120.0) -> None:
    app.send_task(SECURE_SUM_TASK, args=[session_id])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with Session(engine) as session:
            status = session.get(SecureSession, session_id).status
        if status is SecureSessionStatus.summed:
            return
        if status is SecureSessionStatus.failed:
            raise SystemExit(f"secure session {session_id} failed, see the worker logs")
        time.sleep(0.5)
    raise SystemExit(f"secure session {session_id} was not summed within {timeout:.0f}s")


def run_session(app, base: str, key: str, weights_id: int, seats: list[Seat],
                min_members: int) -> SessionResult:
    session_id = None
    user_ids = {}
    for seat in seats:
        resp = join(base, seat.token, key, weights_id, seat.pk)
        if session_id is not None and resp["session_id"] != session_id:
            raise SystemExit("joins were split across sessions; lower the session size")
        session_id = resp["session_id"]
        user_ids[seat.user] = resp["user_id"]
    n = seal_session(session_id, min_members)

    desc = get_descriptor(base, seats[0].token, session_id)
    scale, clip = desc["scale"], desc["clip_bound"]
    roster = [(e["user_id"], base64.b64decode(e["ka_public_key"])) for e in desc["roster"]]
    masked, plain = [], []
    for seat in seats:
        q = quantize(seat.delta, clip, scale)
        y = mask_vector(q, user_ids[seat.user], roster, seat.sk, session_id)
        submit_masked(base, seat.token, session_id, y.astype("<u4").tobytes())
        masked.append(y)
        plain.append(q)

    mean = dequantize(ring_sum(plain), scale, n)
    residual = float(np.max(np.abs(dequantize(ring_sum(masked), scale, n) - mean)))
    sum_session(app, session_id)
    return SessionResult(session_id, n, scale, mean, residual)
