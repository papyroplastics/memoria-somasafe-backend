"""SQLModel tables for the gateway.

Model/quantization blobs live in their own tables keyed by the owning row, so a
row read constantly (e.g. GlobalWeights) never drags a .tflite along; Blobs are
stored and served zstd-compressed; DB signatures cover the *raw* bytes, so
compression happens after signing and the client verifies after decompressing.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DDL, JSON, BigInteger, Column, UniqueConstraint, event
from sqlalchemy.orm import defer
from sqlmodel import Field, Session, SQLModel, create_engine, select

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


class SubmissionType(str, Enum):
    """How a version's weight updates are uploaded and aggregated. ``raw`` and
    ``quantize`` share a dense LE-float32 body (``quantize`` also accepts the raw
    path); ``secure`` carries masked ring elements aggregated only inside a sealed
    ``SecureRound`` (see shared/docs/secure-aggregation.md)."""

    raw = "raw"
    quantize = "quantize"
    secure = "secure"


class Artifact(str, Enum):
    """A serving artifact baked from a GlobalWeights snapshot: the
    LiteRT-trainable .tflite the app fine-tunes, and the signed int8 .tflite the
    firmware runs. Doubles as the path parameter of /model/download/{artifact}."""

    trainable = "trainable"
    quantized = "quantized"


class SecureRoundStatus(str, Enum):
    """Lifecycle of one secure-aggregation round. A round is a first-class object
    (unlike the implicit window the dense paths aggregate over) because masking
    requires the cohort and its public keys to be frozen before anyone masks."""

    open = "open"              # accepting members (roster still mutable)
    sealed = "sealed"          # roster + keys frozen; accepting masked vectors
    aggregated = "aggregated"  # summed, dequantized, baked into new weights
    failed = "failed"          # a member never submitted — masks can't cancel


class IntPKModel(SQLModel):
    """Base for tables with an autoincrement integer primary key. The default of
    None is what SQLAlchemy fills in on flush, but a PK is never actually null
    once persisted, so the annotation is int (not int | None) to spare callers a
    needless narrowing at every query site."""

    id: int = Field(default=None, primary_key=True)  # type: ignore[assignment]


class User(IntPKModel, table=True):
    """An account allowed to talk to the gateway. Created by the seed script
    (no public registration); the password is argon2-hashed (see api.routes.auth)."""

    username: str = Field(unique=True, index=True)
    email: str | None = None
    hashed_password: str
    disabled: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class AuthSession(SQLModel, table=True):
    """A login session's long-lived half: only the refresh token (opaque, stored
    as sha256 so a row can be revoked to kill refresh at once). The short-lived
    access token is validated against Redis instead (api.lib.session)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    refresh_hash: str = Field(unique=True, index=True)
    refresh_expires_at: datetime
    revoked: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class Device(SQLModel, table=True):
    """A physical SomaSafe device with its factory ECDSA P-256 key. Seeded
    ownerless; ``owner_id`` is set once a user attests ownership (api.routes.device)
    and ``last_attested_at`` gates changes to once per DEVICE_ATTEST_COOLDOWN_SECONDS."""

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
    firmware_id: int | None = None


class ModelVersion(IntPKModel, table=True):
    """One published version of a model. ``version`` is the hand-bumped registry
    integer, ``fingerprint`` its arch tripwire; only the newest version per model
    accepts submissions. ``contract_version`` fixes how the device feeds it."""

    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    version: int
    fingerprint: str
    weight_count: int
    contract_version: int
    submission_type: SubmissionType
    norm_params: bytes         # the version's z-score params (LE float32)
    min_app_version: str
    created_at: datetime = Field(default_factory=utcnow)

    __table_args__ = (UniqueConstraint("model_key", "version"),)


class GlobalWeights(IntPKModel, table=True):
    """A snapshot of a version's global weights, its serving artifacts kept as
    keyed WeightsArtifact rows. The active weights are the latest **valid** row
    (``created_at`` gates client re-pulls); ``valid`` is also a kill switch."""

    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    version_id: int = Field(foreign_key="modelversion.id", index=True)
    weights: bytes             # packed float32 (np.float32 .tobytes())
    valid: bool = True
    created_at: datetime = Field(default_factory=utcnow, index=True)


class WeightsArtifact(SQLModel, table=True):
    """One serving artifact of a snapshot; row existence *is* the download route's
    presence check (a snapshot may carry a trainable and no quantized artifact).
    ``signature`` is the server's ECDSA over its canonical model bytes (ml.payload)."""

    weights_id: int = Field(foreign_key="globalweights.id", primary_key=True,
                            ondelete="CASCADE")
    artifact: Artifact = Field(primary_key=True)
    data: bytes                # zstd-compressed .tflite, served as stored
    signature: bytes           # DER ECDSA over the canonical model bytes


class ClientDeltaSubmission(IntPKModel, table=True):
    """A client-uploaded weight *delta* (Δ = local − global). ``base_weights_id``
    is the GlobalWeights it trained from — aggregation matches on it so only
    same-base deltas mix; ``valid`` is the cached structural-check verdict."""

    user_id: int = Field(foreign_key="user.id", index=True)
    base_weights_id: int = Field(foreign_key="globalweights.id", index=True)
    deltas: bytes              # packed float32 delta (np.float32 .tobytes())
    weight_count: int
    valid: bool | None = None
    created_at: datetime = Field(default_factory=utcnow)


class SecureRound(IntPKModel, table=True):
    """One secure-aggregation round, pinned to the ``base_weights`` every member
    trains against. ``member_count`` and ``scale`` are null until seal; ``clip_bound``
    bounds per-coordinate influence (see shared/docs/secure-aggregation.md)."""

    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    version_id: int = Field(foreign_key="modelversion.id", index=True)
    base_weights_id: int = Field(foreign_key="globalweights.id")
    status: SecureRoundStatus = Field(default=SecureRoundStatus.open, index=True)
    clip_bound: float
    member_count: int | None = None
    scale: int | None = Field(default=None, sa_column=Column(BigInteger))
    created_at: datetime = Field(default_factory=utcnow)
    sealed_at: datetime | None = None
    finished_at: datetime | None = None


class SecureRoundMember(SQLModel, table=True):
    """A client's seat in a round; the (round_id, user_id) PK enforces one
    submission per client (a second masked vector would leak a difference of two).
    ``masked`` is the submitted vector (m LE uint32 ring elements), set once."""

    round_id: int = Field(foreign_key="secureround.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    ka_public_key: bytes       # 65-byte uncompressed P-256 point, snapshot at join
    masked: bytes | None = None
    submitted_at: datetime | None = None


class QuantizationJob(SQLModel, table=True):
    """Tracks one quantization request end to end; the int8 .tflite is a keyed
    QuantizationResult row and ``signature`` the server's ECDSA over it. Both are
    dropped by the cleanup sweep once served (after a grace) or expired."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    submission_id: int = Field(foreign_key="clientdeltasubmission.id")
    model_key: str
    status: JobStatus = Field(default=JobStatus.pending)
    signature: bytes | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    served_at: datetime | None = None


class QuantizationResult(SQLModel, table=True):
    """A job's personalized int8 .tflite, held until the cleanup sweep drops it.
    Kept out of QuantizationJob so the sweep can select expiring jobs without
    loading the very blobs it is about to delete."""

    job_id: uuid.UUID = Field(foreign_key="quantizationjob.id", primary_key=True,
                              ondelete="CASCADE")
    data: bytes                # zstd-compressed int8 .tflite, served as stored


class Firmware(IntPKModel, table=True):
    """A published firmware build served by the /ota routes. ``interface_version``
    is the BLE contract an app must share, ``supported_contracts`` the model
    contracts it consumes, ``signature`` the ECDSA over the raw image (``data``)."""

    version: str = Field(unique=True, index=True)  # arbitrary <=32-byte build string
    interface_version: int = Field(index=True)
    supported_contracts: list[int] = Field(sa_column=Column(JSON))
    size: int                                      # raw (uncompressed) image size
    signature: bytes                               # DER ECDSA over the raw image
    data: bytes                                    # zstd-compressed raw image, served as stored
    created_at: datetime = Field(default_factory=utcnow)


# The blob columns already hold zstd output, so TOAST's compression pass can only
# burn CPU failing to shrink them: keep them out of line and uncompressed.
for _blob_model in (WeightsArtifact, QuantizationResult, Firmware):
    event.listen(
        _blob_model.__table__, "after_create",  # type: ignore[attr-defined]
        DDL(f"ALTER TABLE {_blob_model.__tablename__} "
            f"ALTER COLUMN data SET STORAGE EXTERNAL"))


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


def get_weights_artifact(session: Session, weights_id: int,
                         artifact: Artifact) -> WeightsArtifact | None:
    """One baked artifact of a snapshot, or ``None`` if it was never exported."""
    return session.get(WeightsArtifact, (weights_id, artifact))


def get_latest_weights(session: Session, key: str) -> GlobalWeights | None:
    """The model's active weights: the newest valid row of its latest version.
    ``None`` if the model is unknown or has no usable weights yet."""
    latest = get_latest_version(session, key)
    if latest is None:
        return None
    return get_version_weights(session, latest.id)


def get_open_round(session: Session, key: str) -> "SecureRound | None":
    """The model's current ``open`` secure round, if any — the one a joining
    client is added to. ``None`` once it seals (a new round opens on the next
    join), so a fresh round always pins the then-active base weights."""
    return session.exec(
        select(SecureRound)
        .where(SecureRound.model_key == key,
               SecureRound.status == SecureRoundStatus.open)
        .order_by(SecureRound.created_at.desc())  # type: ignore[attr-defined]
    ).first()


def list_firmware(session: Session, interface_version: int) -> list[Firmware]:
    """Every published firmware build for a BLE interface version, newest first.
    The image blob is deferred — the listing never reads it."""
    return list(session.exec(
        select(Firmware)
        .where(Firmware.interface_version == interface_version)
        .order_by(Firmware.created_at.desc())  # type: ignore[attr-defined]
        .options(defer(Firmware.data))  # type: ignore[arg-type]
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
