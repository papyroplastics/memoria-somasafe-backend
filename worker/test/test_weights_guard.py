import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from common.db import GlobalWeights, engine


def _add(session: Session, seeded, parent_id: int | None, valid: bool = True) -> GlobalWeights:
    model_key, version_id, _ = seeded
    row = GlobalWeights(model_key=model_key, version_id=version_id,
                        parent_weights_id=parent_id, weights=b"w", valid=valid)
    session.add(row)
    session.flush()
    return row


def test_second_valid_child_rejected(seeded):
    with Session(engine) as session:
        parent = _add(session, seeded, None, valid=False)
        _add(session, seeded, parent.id)
        with pytest.raises(IntegrityError):
            _add(session, seeded, parent.id)
        session.rollback()


def test_revoked_child_frees_parent(seeded):
    with Session(engine) as session:
        parent = _add(session, seeded, None, valid=False)
        child = _add(session, seeded, parent.id)
        child.valid = False
        session.add(child)
        session.flush()
        _add(session, seeded, parent.id)
        session.rollback()


def test_seeded_rows_do_not_collide(seeded):
    with Session(engine) as session:
        _add(session, seeded, None)
        _add(session, seeded, None)
        session.rollback()
