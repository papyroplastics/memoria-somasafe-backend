import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlmodel import Field, Session, SQLModel, create_engine, select

from common.config import DATABASE_URL


def utcnow() -> datetime:
    # Naive UTC to match the default TIMESTAMP WITHOUT TIME ZONE columns.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ModelPurpose(str, Enum):
    """TF-free model role, shared by the code registry (ml.model_list) and the
    ModelDefinition table the registry is projected into by the seed script."""

    train_only = "train-only"
    embed_infer = "embed-infer"
    app_infer = "app-infer"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    expired = "expired"


class User(SQLModel, table=True):
    """An account allowed to talk to the gateway. Created by the seed script
    (no public registration); the password is argon2-hashed (see api.routes.auth)."""

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


class Device(SQLModel, table=True):
    """A physical SomaSafe device, provisioned with a factory ECDSA P-256 key.
    Seeded ownerless by scripts.db_seed from a factory NVS image; ``owner_id`` is
    set once a user attests ownership (see api.routes.device). ``last_attested_at``
    gates ownership changes to once per DEVICE_ATTEST_COOLDOWN_SECONDS."""

    serial: str = Field(primary_key=True)
    public_key: bytes          # 65-byte uncompressed P-256 point (0x04 || X || Y)
    owner_id: int | None = Field(default=None, foreign_key="user.id", index=True)
    last_attested_at: datetime | None = None


class ModelFingerprint(SQLModel, table=True):
    """An architecture identity. ``fingerprint`` is derived from the model's
    trainable-variable layout (TrainableModel.arch_fingerprint) and is the
    weight-compatibility boundary: parameters are only interchangeable within
    one fingerprint. ``display_version`` is the human-facing label from the code
    registry (ml.model_list) — informational, not a source of truth."""

    fingerprint: str = Field(primary_key=True)
    display_version: str
    param_count: int
    created_at: datetime = Field(default_factory=utcnow)


class ModelDefinition(SQLModel, table=True):
    """Per-model metadata registry, served by the gateway. Seeded by
    scripts.db_seed from ml.model_list; ``fingerprint`` is the architecture
    currently deployed in code (clients reset their local model when it changes,
    which is the only thing that invalidates a federated epoch)."""

    key: str = Field(primary_key=True)
    name: str
    last_updated: datetime = Field(default_factory=utcnow)
    purpose: ModelPurpose
    firmware_id: int | None = None
    app_version: str
    fingerprint: str = Field(foreign_key="modelfingerprint.fingerprint", index=True)


class GlobalWeights(SQLModel, table=True):
    """A snapshot of a model's global parameters. Seeded from the trained tflite
    and appended to by aggregation. The active weights of a model are the latest
    **valid** row matching its current ``fingerprint``; ``created_at`` is the
    weight version clients compare against to decide when to re-pull weights.
    ``valid`` is a hand-operated kill switch: set it to false to roll back an
    aggregation round that made the model worse. ``mse_threshold`` is the
    allowed submission error computed by that round (None on seeded rows, which
    skips the check in weight validation)."""

    id: int | None = Field(default=None, primary_key=True)
    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    fingerprint: str = Field(foreign_key="modelfingerprint.fingerprint", index=True)
    parameters: bytes          # packed float32 (np.float32 .tobytes())
    param_count: int
    valid: bool = True
    mse_threshold: float | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)


class WeightSubmission(SQLModel, table=True):
    """A client-uploaded weight update. Persisted indefinitely: besides feeding
    quantization, these rows are the substrate for federated aggregation
    (worker.tasks.federated_aggregation). ``base_weights_id`` is the
    GlobalWeights snapshot the client trained from; ``fingerprint`` (its
    architecture) is denormalized off it so aggregation can filter without a
    join and never mixes incompatible updates. ``valid`` is the cached
    weight-validation verdict — None until validated; normally set by
    quantize_submission, and by aggregation for rows that never went through
    a quantization job."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    model_key: str
    base_weights_id: int = Field(foreign_key="globalweights.id", index=True)
    fingerprint: str = Field(foreign_key="modelfingerprint.fingerprint", index=True)
    parameters: bytes          # packed float32 (np.float32 .tobytes())
    param_count: int
    valid: bool | None = None
    created_at: datetime = Field(default_factory=utcnow)


class QuantizationJob(SQLModel, table=True):
    """Tracks one quantization request end to end. ``result`` holds the int8
    .tflite bytes and is the only ephemeral field — it is nulled by the cleanup
    sweep once served (after a grace period) or once expired (unclaimed)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    submission_id: int = Field(foreign_key="weightsubmission.id")
    model_key: str
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


def get_model_def(session: Session, key: str) -> ModelDefinition | None:
    return session.get(ModelDefinition, key)


def get_latest_weights(session: Session, key: str) -> GlobalWeights | None:
    """The model's active weights: the newest **valid** GlobalWeights row
    matching its current architecture fingerprint. ``None`` if the model is
    unknown or has no compatible weights yet (e.g. right after an architecture
    change, or every compatible row was invalidated by hand)."""
    meta = session.get(ModelDefinition, key)
    if meta is None:
        return None
    return session.exec(
        select(GlobalWeights)
        .where(GlobalWeights.model_key == key,
               GlobalWeights.fingerprint == meta.fingerprint,
               GlobalWeights.valid == True)  # noqa: E712 — SQL expression
        .order_by(GlobalWeights.created_at.desc())  # type: ignore[attr-defined]
    ).first()


def user_owns_device(session: Session, user_id: int) -> bool:
    """True if the user has attested ownership of at least one device."""
    return session.exec(
        select(Device).where(Device.owner_id == user_id)
    ).first() is not None
