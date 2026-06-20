"""Seed the database with default rows: the model registry and a default user.

Idempotent — run it after the services are up to bootstrap a fresh database:

    uv run -m scripts.seed
"""

import datetime

from sqlmodel import Session, select

from api.auth import hash_password
from common.config import SEED_EMAIL, SEED_PASSWORD, SEED_USER
from common.db import ModelDefinition, User, engine, init_db
from common.models import ModelPurpose

# Default model registry (previously the static dict in common.models).
MODEL_DEFS = [
    ModelDefinition(
        key="feature-mlp",
        name="Feature-based MLP",
        last_updated=datetime.datetime(2026, 6, 1),
        purpose=ModelPurpose.train_only,
        firmware_id=None,
        app_version="1.0.0",
        model_id=1,
    ),
]


def seed_models(session: Session) -> None:
    for model in MODEL_DEFS:
        if session.get(ModelDefinition, model.key) is None:
            session.add(model)
            print(f"  + model '{model.key}' v{model.model_id}")
    session.commit()


def seed_users(session: Session) -> None:
    existing = session.exec(select(User).where(User.username == SEED_USER)).first()
    if existing is None:
        session.add(User(
            username=SEED_USER,
            email=SEED_EMAIL,
            hashed_password=hash_password(SEED_PASSWORD),
        ))
        session.commit()
        print(f"  + user '{SEED_USER}'")


def main() -> None:
    init_db()
    with Session(engine) as session:
        seed_models(session)
        seed_users(session)
    print("Seed complete.")


if __name__ == "__main__":
    main()
