import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import JSON, Column, UniqueConstraint
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


class ModelDefinition(SQLModel, table=True):
    """Per-model identity registry, served by the gateway. Seeded by
    scripts.seed_db from ml.model_list; everything that varies over a model's
    lifetime lives on its ModelVersion rows."""

    key: str = Field(primary_key=True)
    name: str
    last_updated: datetime = Field(default_factory=utcnow)
    purpose: ModelPurpose
    firmware_id: int | None = None


class ModelVersion(SQLModel, table=True):
    """One published version of a model. ``version`` is the hand-bumped integer
    from the code registry (ml.model_list); ``fingerprint`` (Trainer.arch_fingerprint,
    a hash of the trainable-variable layout plus the baked normalization params) is
    the tripwire the seed script checks it against — a moved fingerprint without a
    version bump aborts the seed. Only the newest version per model accepts weight
    submissions; older versions are frozen but still served. ``min_app_version`` is
    the oldest app that can use the version; ``contract_version`` fixes how the
    device feeds the model (norm_params layout + I/O signatures, see ml.payload)."""

    id: int | None = Field(default=None, primary_key=True)
    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    version: int
    fingerprint: str
    param_count: int
    contract_version: int
    norm_params: bytes         # the version's z-score params (LE float32)
    min_app_version: str
    created_at: datetime = Field(default_factory=utcnow)

    __table_args__ = (UniqueConstraint("model_key", "version"),)


class GlobalWeights(SQLModel, table=True):
    """A snapshot of a model version's global parameters, plus the serving
    artifacts baked from them. Seeded from the trained tflite files and appended
    to by aggregation, which re-exports both artifacts (and signs the quantized
    one, see ml.payload) each round. The active weights of a version are its
    latest **valid** row; ``created_at`` is the weight version clients compare
    against to decide when to re-pull. ``valid`` doubles as a hand-operated kill
    switch (flip it to roll back a bad round — the previous row's artifacts come
    back with it) and is set to false by aggregation itself when an artifact
    export fails. ``mse_threshold`` is the allowed submission error computed by
    that round (None on seeded rows, which skips the check in weight validation)."""

    id: int | None = Field(default=None, primary_key=True)
    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    version_id: int = Field(foreign_key="modelversion.id", index=True)
    parameters: bytes          # packed float32 (np.float32 .tobytes())
    param_count: int
    valid: bool = True
    mse_threshold: float | None = None
    trainable_artifact: bytes | None = None   # LiteRT-trainable .tflite with these weights
    quantized_artifact: bytes | None = None   # int8 .tflite with these weights
    artifact_signature: bytes | None = None   # DER ECDSA over the canonical model bytes
    created_at: datetime = Field(default_factory=utcnow, index=True)


class WeightSubmission(SQLModel, table=True):
    """A client-uploaded weight update. Persisted indefinitely: besides feeding
    quantization, these rows are the substrate for federated aggregation
    (worker.tasks.federated_aggregation). ``base_weights_id`` is the
    GlobalWeights snapshot the client trained from; ``version_id`` (its model
    version) is denormalized off it so aggregation can filter without a join and
    never mixes incompatible updates. ``valid`` is the cached weight-validation
    verdict — None until validated; set by the quantize/validate tasks, and by
    aggregation for rows neither got to. The verdict is never surfaced to the
    client (a Byzantine client should not learn its update was filtered)."""

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    model_key: str
    base_weights_id: int = Field(foreign_key="globalweights.id", index=True)
    version_id: int = Field(foreign_key="modelversion.id", index=True)
    parameters: bytes          # packed float32 (np.float32 .tobytes())
    param_count: int
    valid: bool | None = None
    created_at: datetime = Field(default_factory=utcnow)


class QuantizationJob(SQLModel, table=True):
    """Tracks one quantization request end to end. ``result`` holds the int8
    .tflite bytes and ``signature`` the server's ECDSA over its canonical model
    bytes (ml.payload); both are ephemeral — nulled by the cleanup sweep once
    served (after a grace period) or once expired (unclaimed)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    submission_id: int = Field(foreign_key="weightsubmission.id")
    model_key: str
    status: JobStatus = Field(default=JobStatus.pending)
    result: bytes | None = None
    signature: bytes | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    served_at: datetime | None = None


class Firmware(SQLModel, table=True):
    """A published firmware build, seeded from a `shared/gen/firmware/{version}`
    export (`firmware/scripts/export_image.py`) and served by the /ota routes.
    ``interface_version`` is the BLE contract an app build must share to talk to
    it; ``supported_contracts`` are the model contract versions it can consume;
    ``signature`` is the server's ECDSA over the raw image bytes, verified by
    the device against its factory srv_pub before booting the image."""

    id: int | None = Field(default=None, primary_key=True)
    version: str = Field(unique=True, index=True)  # arbitrary <=32-byte build string
    interface_version: int = Field(index=True)
    supported_contracts: list[int] = Field(sa_column=Column(JSON))
    blob: bytes
    signature: bytes | None = None                 # DER ECDSA over the raw image
    created_at: datetime = Field(default_factory=utcnow)


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


def get_latest_version(session: Session, key: str) -> ModelVersion | None:
    """The model's newest published version — the only one accepting submissions."""
    return session.exec(
        select(ModelVersion)
        .where(ModelVersion.model_key == key)
        .order_by(ModelVersion.version.desc())  # type: ignore[attr-defined]
    ).first()


def get_version_weights(session: Session, version_id: int) -> GlobalWeights | None:
    """A version's active weights: its newest **valid** GlobalWeights row.
    ``None`` if every row was invalidated (by hand or by a failed export)."""
    return session.exec(
        select(GlobalWeights)
        .where(GlobalWeights.version_id == version_id,
               GlobalWeights.valid == True)  # noqa: E712 — SQL expression
        .order_by(GlobalWeights.created_at.desc())  # type: ignore[attr-defined]
    ).first()


def get_latest_weights(session: Session, key: str) -> GlobalWeights | None:
    """The model's active weights: the newest valid row of its latest version.
    ``None`` if the model is unknown or has no usable weights yet."""
    latest = get_latest_version(session, key)
    if latest is None:
        return None
    return get_version_weights(session, latest.id)


def list_firmware(session: Session, interface_version: int) -> list[Firmware]:
    """Every published firmware build for a BLE interface version, newest first."""
    return list(session.exec(
        select(Firmware)
        .where(Firmware.interface_version == interface_version)
        .order_by(Firmware.created_at.desc())  # type: ignore[attr-defined]
    ).all())


def get_firmware(session: Session, interface_version: int,
                 version: str) -> Firmware | None:
    return session.exec(
        select(Firmware).where(Firmware.interface_version == interface_version,
                               Firmware.version == version)
    ).first()


def user_owns_device(session: Session, user_id: int) -> bool:
    """True if the user has attested ownership of at least one device."""
    return session.exec(
        select(Device).where(Device.owner_id == user_id)
    ).first() is not None
