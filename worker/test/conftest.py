import pytest
from sqlmodel import Session, select

from common.db import (
    ModelVersion,
    SecureSession,
    SecureSessionMember,
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
def open_session(seeded):
    model_key, version_id, weights_id = seeded
    with Session(engine) as session:
        row = SecureSession(model_key=model_key, version_id=version_id,
                            base_weights_id=weights_id, clip_bound=1.0)
        session.add(row)
        session.commit()
        session_id = row.id
    yield session_id
    with Session(engine) as session:
        for member in session.exec(select(SecureSessionMember)
                                   .where(SecureSessionMember.session_id == session_id)):
            session.delete(member)
        session.flush()
        row = session.get(SecureSession, session_id)
        if row is not None:
            session.delete(row)
        session.commit()
