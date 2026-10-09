import threading
import uuid

from sqlmodel import Session

from common.db import (
    JobStatus,
    QuantizationJob,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    engine,
    seal_session,
)


def _status(session_id: int) -> SecureSessionStatus:
    with Session(engine) as session:
        return session.get(SecureSession, session_id).status


def test_claim_once(open_session):
    assert SecureSession.claim(open_session, SecureSessionStatus.open,
                               SecureSessionStatus.sealed)
    assert not SecureSession.claim(open_session, SecureSessionStatus.open,
                                   SecureSessionStatus.sealed)
    assert _status(open_session) is SecureSessionStatus.sealed


def test_claim_from_any_of(open_session):
    assert SecureSession.claim(open_session, (SecureSessionStatus.sealed, SecureSessionStatus.open),
                             SecureSessionStatus.failed)
    assert _status(open_session) is SecureSessionStatus.failed


def test_claim_uuid_pk_missing_row():
    assert not QuantizationJob.claim(uuid.uuid4(), JobStatus.pending, JobStatus.running)


def test_transition_rolled_back(open_session):
    with Session(engine) as session:
        assert SecureSession.transition(session, open_session, SecureSessionStatus.open,
                                      SecureSessionStatus.sealed)
        session.rollback()
    assert _status(open_session) is SecureSessionStatus.open


def test_concurrent_claims_single_winner(open_session):
    barrier = threading.Barrier(8)
    wins = []

    def contend():
        barrier.wait()
        wins.append(SecureSession.claim(open_session, SecureSessionStatus.open,
                                      SecureSessionStatus.sealed))

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(wins) == 1


def test_seal_session(open_session, seeded, test_user_ids):
    _, _, weights_id = seeded
    with Session(engine) as session:
        for user_id in test_user_ids:
            session.add(SecureSessionMember(session_id=open_session, user_id=user_id,
                                            base_weights_id=weights_id,
                                            ka_public_key=b"\x04" + bytes(64)))
        session.commit()

    with Session(engine) as session:
        assert seal_session(session, open_session) == 3
        session.commit()
    with Session(engine) as session:
        row = session.get(SecureSession, open_session)
        assert row.status is SecureSessionStatus.sealed
        assert row.member_count == 3 and row.scale and row.sealed_at
        assert seal_session(session, open_session) is None
