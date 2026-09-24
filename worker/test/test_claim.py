import threading
import uuid

from sqlmodel import Session

from common.db import (
    JobStatus,
    QuantizationJob,
    SecureRound,
    SecureRoundMember,
    SecureRoundStatus,
    engine,
)
from common.secure_round import seal_round


def _status(round_id: int) -> SecureRoundStatus:
    with Session(engine) as session:
        return session.get(SecureRound, round_id).status


def test_claim_once(open_round):
    assert SecureRound.claim(open_round, SecureRoundStatus.open, SecureRoundStatus.sealed)
    assert not SecureRound.claim(open_round, SecureRoundStatus.open, SecureRoundStatus.sealed)
    assert _status(open_round) is SecureRoundStatus.sealed


def test_claim_from_any_of(open_round):
    assert SecureRound.claim(open_round, (SecureRoundStatus.sealed, SecureRoundStatus.open),
                             SecureRoundStatus.failed, error="x")
    assert _status(open_round) is SecureRoundStatus.failed


def test_claim_uuid_pk_missing_row():
    assert not QuantizationJob.claim(uuid.uuid4(), JobStatus.pending, JobStatus.running)


def test_transition_rolled_back(open_round):
    with Session(engine) as session:
        assert SecureRound.transition(session, open_round, SecureRoundStatus.open,
                                      SecureRoundStatus.sealed)
        session.rollback()
    assert _status(open_round) is SecureRoundStatus.open


def test_concurrent_claims_single_winner(open_round):
    barrier = threading.Barrier(8)
    wins = []

    def contend():
        barrier.wait()
        wins.append(SecureRound.claim(open_round, SecureRoundStatus.open,
                                      SecureRoundStatus.sealed))

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(wins) == 1


def test_seal_round(open_round, test_user_ids):
    with Session(engine) as session:
        for user_id in test_user_ids:
            session.add(SecureRoundMember(round_id=open_round, user_id=user_id,
                                          ka_public_key=b"\x04" + bytes(64)))
        session.commit()

    with Session(engine) as session:
        assert seal_round(session, open_round) == 3
        session.commit()
    with Session(engine) as session:
        round = session.get(SecureRound, open_round)
        assert round.status is SecureRoundStatus.sealed
        assert round.member_count == 3 and round.scale and round.sealed_at
        assert seal_round(session, open_round) is None
