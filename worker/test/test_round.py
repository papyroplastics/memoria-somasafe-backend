from datetime import timedelta

import pytest
from sqlalchemy import delete
from sqlmodel import Session, select

from common.celery_tasks import FED_AGG_TASK
from common.db import (
    ClientDeltaSubmission,
    GlobalWeights,
    JobStatus,
    ModelVersion,
    QuantizationJob,
    SecurePartial,
    SecureSession,
    SecureSessionMember,
    SecureSessionStatus,
    User,
    count_summed_members,
    engine,
    utcnow,
)
from common.secure_agg import compute_scale
from worker.metrics import Timer
from worker.tasks import aggregation, secure
from worker.tasks.cleanup import _clear_payloads


@pytest.fixture
def stale_base(seeded):
    model_key, version_id, _ = seeded
    with Session(engine) as session:
        row = GlobalWeights(model_key=model_key, version_id=version_id, weights=b"w",
                            valid=False)
        session.add(row)
        session.commit()
        base_id = row.id
    yield base_id
    with Session(engine) as session:
        submissions = select(ClientDeltaSubmission.id).where(
            ClientDeltaSubmission.base_weights_id == base_id)
        sessions = select(SecureSession.id).where(SecureSession.base_weights_id == base_id)
        session.execute(delete(QuantizationJob).where(QuantizationJob.submission_id.in_(submissions)))  # type: ignore[attr-defined]
        session.execute(delete(ClientDeltaSubmission).where(ClientDeltaSubmission.id.in_(submissions)))  # type: ignore[attr-defined]
        session.execute(delete(SecurePartial).where(SecurePartial.session_id.in_(sessions)))  # type: ignore[attr-defined]
        session.execute(delete(SecureSessionMember).where(SecureSessionMember.session_id.in_(sessions)))  # type: ignore[attr-defined]
        session.execute(delete(SecureSession).where(SecureSession.id.in_(sessions)))  # type: ignore[attr-defined]
        session.execute(delete(GlobalWeights).where(GlobalWeights.id == base_id))
        session.commit()


@pytest.fixture
def active_base(seeded):
    _, version_id, weights_id = seeded
    with Session(engine) as session:
        original = session.get(ModelVersion, version_id).round_limit
        floor = session.exec(select(SecureSession.id).order_by(SecureSession.id.desc())).first() or 0

    def set_limit(value: int) -> None:
        with Session(engine) as session:
            version = session.get(ModelVersion, version_id)
            version.round_limit = value
            session.add(version)
            session.commit()

    yield weights_id, set_limit
    set_limit(original)
    with Session(engine) as session:
        session.get(GlobalWeights, weights_id).closed = False
        sessions = select(SecureSession.id).where(SecureSession.id > floor)
        session.execute(delete(SecurePartial).where(SecurePartial.session_id.in_(sessions)))  # type: ignore[attr-defined]
        session.execute(delete(SecureSessionMember).where(SecureSessionMember.session_id.in_(sessions)))  # type: ignore[attr-defined]
        session.execute(delete(SecureSession).where(SecureSession.id.in_(sessions)))  # type: ignore[attr-defined]
        session.commit()


def _submission(session: Session, base_id: int, user_id: int, age: timedelta = timedelta()) -> int:
    row = ClientDeltaSubmission(user_id=user_id, base_weights_id=base_id, deltas=b"\x00" * 8,
                                weight_count=2, created_at=utcnow() - age)
    session.add(row)
    session.flush()
    return row.id


def _deltas(submission_id: int) -> bytes | None:
    with Session(engine) as session:
        return session.get(ClientDeltaSubmission, submission_id).deltas


def _session(session: Session, seeded, base_id: int, status: SecureSessionStatus,
             user_ids: list[int], masked: bytes | None = None) -> int:
    model_key, version_id, _ = seeded
    row = SecureSession(model_key=model_key, version_id=version_id, base_weights_id=base_id,
                        clip_bound=1.0, status=status, member_count=len(user_ids),
                        finished_at=utcnow())
    session.add(row)
    session.flush()
    for user_id in user_ids:
        session.add(SecureSessionMember(session_id=row.id, user_id=user_id,
                                        base_weights_id=base_id,
                                        ka_public_key=b"\x04" + bytes(64), masked=masked))
    session.flush()
    return row.id


def test_cleanup_clears_finished_rounds(seeded, stale_base, test_user_ids):
    with Session(engine) as session:
        done = _submission(session, stale_base, test_user_ids[0])
        quantizing = _submission(session, stale_base, test_user_ids[1])
        session.add(QuantizationJob(submission_id=quantizing, model_key=seeded[0],
                                    status=JobStatus.pending))
        session_id = _session(session, seeded, stale_base, SecureSessionStatus.summed,
                              test_user_ids, masked=b"\x00" * 8)
        session.add(SecurePartial(session_id=session_id, mean=b"\x00" * 8))
        session.commit()

    with Session(engine) as session:
        _clear_payloads(session, utcnow() - timedelta(days=10))

    assert _deltas(done) is None
    assert _deltas(quantizing) is not None
    with Session(engine) as session:
        assert session.get(SecurePartial, session_id) is None
        members = session.exec(select(SecureSessionMember)
                               .where(SecureSessionMember.session_id == session_id)).all()
        assert len(members) == len(test_user_ids)
        assert all(member.masked is None for member in members)


def test_cleanup_keeps_open_round_until_retention(seeded, test_user_ids):
    _, _, active_id = seeded
    with Session(engine) as session:
        fresh = _submission(session, active_id, test_user_ids[0])
        old = _submission(session, active_id, test_user_ids[1], age=timedelta(days=11))
        session.commit()
    try:
        with Session(engine) as session:
            _clear_payloads(session, utcnow() - timedelta(days=10))
        assert _deltas(fresh) is not None
        assert _deltas(old) is None
    finally:
        with Session(engine) as session:
            session.execute(delete(ClientDeltaSubmission)
                            .where(ClientDeltaSubmission.id.in_([fresh, old])))  # type: ignore[attr-defined]
            session.commit()


def _sealed(seeded, base_id: int, user_ids: list[int]) -> int:
    model_key, version_id, _ = seeded
    with Session(engine) as session:
        weight_count = session.get(ModelVersion, version_id).weight_count
        row = SecureSession(model_key=model_key, version_id=version_id, base_weights_id=base_id,
                            clip_bound=1.0, status=SecureSessionStatus.sealed,
                            member_count=len(user_ids), scale=compute_scale(len(user_ids), 1.0),
                            sealed_at=utcnow())
        session.add(row)
        session.flush()
        for user_id in user_ids:
            session.add(SecureSessionMember(session_id=row.id, user_id=user_id,
                                            base_weights_id=base_id,
                                            ka_public_key=b"\x04" + bytes(64),
                                            masked=bytes(weight_count * 4), submitted_at=utcnow()))
        session.commit()
        return row.id


def _summed_members(base_id: int) -> int:
    with Session(engine) as session:
        return count_summed_members(session, base_id)


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(secure.app, "send_task",
                        lambda name, args=None, **_: calls.append((name, args)))
    return calls


def test_sum_below_limit_keeps_round_open(seeded, active_base, test_user_ids, sent):
    base_id, set_limit = active_base
    set_limit(_summed_members(base_id) + 10 * len(test_user_ids))
    session_id = _sealed(seeded, base_id, test_user_ids)
    secure.secure_session_sum(session_id)

    with Session(engine) as session:
        assert session.get(SecureSession, session_id).status is SecureSessionStatus.summed
        assert not session.get(GlobalWeights, base_id).closed
    assert sent == []


def test_sum_reaching_limit_closes_round(seeded, active_base, test_user_ids, sent):
    base_id, set_limit = active_base
    set_limit(_summed_members(base_id) + len(test_user_ids) - 1)
    with Session(engine) as session:
        others = session.exec(select(User.id).where(
            User.username.in_(["test_4", "test_5", "test_6"]))).all()  # type: ignore[attr-defined]
    first = _sealed(seeded, base_id, test_user_ids)
    late = _sealed(seeded, base_id, list(others))
    secure.secure_session_sum(first)

    with Session(engine) as session:
        assert session.get(SecureSession, first).status is SecureSessionStatus.summed
        assert session.get(GlobalWeights, base_id).closed
    assert sent == [(FED_AGG_TASK, [seeded[0], base_id])]

    secure.secure_session_sum(late)
    with Session(engine) as session:
        assert session.get(SecureSession, late).status is SecureSessionStatus.failed
        assert session.get(SecurePartial, late) is None
        assert not session.exec(select(SecureSessionMember)
                                .where(SecureSessionMember.session_id == late)).all()
    assert len(sent) == 1


def test_round_below_minimum_reopens(seeded, active_base, monkeypatch):
    base_id, _ = active_base
    calls = []
    real = aggregation._set_closed
    monkeypatch.setattr(aggregation, "_set_closed",
                        lambda weights_id, closed: (calls.append((weights_id, closed)),
                                                    real(weights_id, closed)))
    monkeypatch.setattr(aggregation, "FED_MIN_SUBMISSIONS", 10**9)
    record = aggregation.AggregationRecord(seeded[0])
    timer = Timer(*aggregation.STAGES)

    assert aggregation._read_round(seeded[0], base_id, timer, record) is None
    assert record.outcome is aggregation.AggregationOutcome.skipped_min_submissions
    assert calls == [(base_id, True), (base_id, False)]


def test_round_for_replaced_weights_is_skipped(seeded, active_base, monkeypatch):
    base_id, _ = active_base
    record = aggregation.AggregationRecord(seeded[0])
    timer = Timer(*aggregation.STAGES)

    assert aggregation._read_round(seeded[0], base_id + 10**6, timer, record) is None
    assert record.outcome is aggregation.AggregationOutcome.skipped_stale
    with Session(engine) as session:
        assert not session.get(GlobalWeights, base_id).closed
