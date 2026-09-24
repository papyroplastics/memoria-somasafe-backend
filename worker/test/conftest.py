import pytest
from sqlmodel import Session, select

from common.db import (
    ModelVersion,
    SecureRound,
    SecureRoundMember,
    User,
    engine,
    get_version_weights,
)


@pytest.fixture
def seeded() -> tuple[str, int, int]:
    with Session(engine) as session:
        for version in session.exec(select(ModelVersion)).all():
            weights = get_version_weights(session, version.id)
            if weights is not None:
                return version.model_key, version.id, weights.id
    pytest.skip("seed the database first (make db-seed)")


@pytest.fixture
def test_user_ids() -> list[int]:
    with Session(engine) as session:
        users = session.exec(select(User).where(
            User.username.in_(["test_1", "test_2", "test_3"]))).all()  # type: ignore[attr-defined]
    if len(users) != 3:
        pytest.skip("seed the test users first (make db-seed)")
    return [u.id for u in users]


@pytest.fixture
def open_round(seeded):
    model_key, version_id, weights_id = seeded
    with Session(engine) as session:
        round = SecureRound(model_key=model_key, version_id=version_id,
                            base_weights_id=weights_id, clip_bound=1.0)
        session.add(round)
        session.commit()
        round_id = round.id
    yield round_id
    with Session(engine) as session:
        for member in session.exec(select(SecureRoundMember)
                                   .where(SecureRoundMember.round_id == round_id)):
            session.delete(member)
        session.flush()
        round = session.get(SecureRound, round_id)
        if round is not None:
            session.delete(round)
        session.commit()
