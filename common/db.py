import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, Session, SQLModel, create_engine

from common.config import DATABASE_URL


def utcnow() -> datetime:
    # Naive UTC to match the default TIMESTAMP WITHOUT TIME ZONE columns.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    expired = "expired"


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
