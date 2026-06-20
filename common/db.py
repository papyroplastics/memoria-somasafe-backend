import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, Session, SQLModel, create_engine, select

from common.config import DATABASE_URL
from common.models import ModelPurpose


def utcnow() -> datetime:
    # Naive UTC to match the default TIMESTAMP WITHOUT TIME ZONE columns.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    expired = "expired"


class User(SQLModel, table=True):
    """An account allowed to talk to the gateway. Created by the seed script
    (no public registration); the password is argon2-hashed (see api.auth)."""

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str | None = None
    hashed_password: str
    disabled: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class AuthSession(SQLModel, table=True):
    """A stateful login session. Tokens are opaque random strings; only their
    sha256 is stored, so a row can be revoked to invalidate a session at once."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    access_hash: str = Field(unique=True, index=True)
    refresh_hash: str = Field(unique=True, index=True)
    access_expires_at: datetime
    refresh_expires_at: datetime
    revoked: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: datetime = Field(default_factory=utcnow)


class ModelDefinition(SQLModel, table=True):
    """Per-model metadata registry, served by the gateway and validated against.
    Seeded by scripts.seed; replaces the old static dict in common.models."""

    key: str = Field(primary_key=True)
    name: str
    last_updated: datetime
    purpose: ModelPurpose
    firmware_id: int | None = None
    app_version: str
    model_id: int


class WeightSubmission(SQLModel, table=True):
    """A client-uploaded weight update. Persisted indefinitely: besides feeding
    quantization, these rows are the substrate for the future aggregation step,
    so the worker only ever reads them."""

    id: int | None = Field(default=None, primary_key=True)
    model_key: str
    model_version: int
    parameters: bytes          # packed float32 (np.float32 .tobytes())
    param_count: int
    created_at: datetime = Field(default_factory=utcnow)


class QuantizationJob(SQLModel, table=True):
    """Tracks one quantization request end to end. ``result`` holds the int8
    .tflite bytes and is the only ephemeral field — it is nulled by the cleanup
    sweep once served (after a grace period) or once expired (unclaimed)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    submission_id: int = Field(foreign_key="weightsubmission.id")
    model_key: str
    model_version: int
    status: JobStatus = Field(default=JobStatus.pending)
    result: bytes | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    served_at: datetime | None = None


engine = create_engine(DATABASE_URL)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


def list_model_defs(session: Session) -> list[ModelDefinition]:
    return list(session.exec(select(ModelDefinition)).all())


def get_model_def(session: Session, key: str, version: int) -> ModelDefinition | None:
    """Return the model row only if both the key and the version (model_id) match."""
    meta = session.get(ModelDefinition, key)
    if meta is None or meta.model_id != version:
        return None
    return meta
