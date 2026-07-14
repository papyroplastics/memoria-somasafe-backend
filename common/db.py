import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import JSON, BigInteger, Column, UniqueConstraint
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
    """How a model version's weight updates are uploaded and aggregated. Fixes
    both the submission endpoint a client uses and the aggregation strategy the
    worker applies. ``quantize`` also accepts submissions on the raw path (same
    dense LE-float32 format, less backend work); ``raw`` does not accept the
    quantize path. ``secure`` carries an incompatible body (masked ring elements,
    not float32) and only aggregates inside a sealed ``SecureRound`` — it shares
    nothing with the dense paths (see shared/docs/secure-aggregation.md). Future
    formats (sparse, DP) likewise get their own endpoint."""

    raw = "raw"
    quantize = "quantize"
    secure = "secure"


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
    """A stateful login session's long-lived half. Only the refresh token lives
    here (opaque random string, only its sha256 stored, so a row can be revoked
    to invalidate refresh at once); the short-lived access token is validated
    against Redis instead (api.lib.session) — a Postgres row per request would be
    pure overhead for a token that expires in minutes anyway."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    refresh_hash: str = Field(unique=True, index=True)
    refresh_expires_at: datetime
    revoked: bool = False
    created_at: datetime = Field(default_factory=utcnow)


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
    firmware_id: int | None = None


class ModelVersion(IntPKModel, table=True):
    """One published version of a model. ``version`` is the hand-bumped integer
    from the code registry (ml.model_list); ``fingerprint`` (Trainer.arch_fingerprint,
    a hash of the trainable-variable layout plus the baked normalization params) is
    the tripwire the seed script checks it against — a moved fingerprint without a
    version bump aborts the seed. Only the newest version per model accepts weight
    submissions; older versions are frozen but still served. ``min_app_version`` is
    the oldest app that can use the version; ``contract_version`` fixes how the
    device feeds the model (norm_params layout + I/O signatures, see ml.payload)."""

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
    """A snapshot of a model version's global weights. The trainable/quantized
    serving artifacts baked from these weights live on disk under SERVE_DIR,
    keyed by this row (common.storage), not in the DB; only their signature is
    kept here. Seeded from the trained tflite files and appended to by
    aggregation, which re-exports both artifacts (and signs the quantized one,
    see ml.payload) each round. The active weights of a version are its latest
    **valid** row; ``created_at`` is the weight version clients compare against
    to decide when to re-pull. ``valid`` doubles as a hand-operated kill switch
    (flip it to roll back a bad round — the previous row's artifacts come back
    with it) and is set to false by aggregation itself when an artifact export
    fails. ``mse_threshold`` is the allowed submission error computed by that
    round (None on seeded rows, which skips the check in weight validation)."""

    model_key: str = Field(foreign_key="modeldefinition.key", index=True)
    version_id: int = Field(foreign_key="modelversion.id", index=True)
    weights: bytes             # packed float32 (np.float32 .tobytes())
    weight_count: int
    valid: bool = True
    mse_threshold: float | None = None
    artifact_signature: bytes | None = None   # DER ECDSA over the canonical model bytes
    created_at: datetime = Field(default_factory=utcnow, index=True)


class ClientDeltaSubmission(IntPKModel, table=True):
    """A client-uploaded weight *delta* — Δ = local_weights − global_weights, the
    change the client's local training produced against the snapshot it trained
    from. Persisted indefinitely: besides feeding quantization, these rows are the
    substrate for federated aggregation (worker.tasks.federated_aggregation), which
    averages the deltas and adds the mean onto the reference global weights.
    ``base_weights_id`` is the GlobalWeights snapshot the delta is relative to (and
    the client trained from); ``version_id`` (its model version) is denormalized off
    it so aggregation can filter without a join and never mixes incompatible updates.
    ``valid`` is the cached weight-validation verdict — None until validated; set by
    the quantize/validate tasks, and by aggregation for rows neither got to. The
    verdict is never surfaced to the client (a Byzantine client should not learn its
    update was filtered)."""

    user_id: int = Field(foreign_key="user.id", index=True)
    model_key: str
    base_weights_id: int = Field(foreign_key="globalweights.id", index=True)
    version_id: int = Field(foreign_key="modelversion.id", index=True)
    deltas: bytes              # packed float32 delta (np.float32 .tobytes())
    weight_count: int
    valid: bool | None = None
    created_at: datetime = Field(default_factory=utcnow)


class SecureRound(IntPKModel, table=True):
    """One secure-aggregation round for a model version. Created ``open`` by the
    first client to join (pinned to the version's active ``base_weights`` — the W
    every member trains against), sealed once the cohort is fixed, then consumed
    by ``worker.tasks.secure_aggregation``. ``member_count`` (n) and ``scale`` (the
    fixed-point S = floor(2^31 / (n * clip_bound))) are null until seal, when the
    roster size is known. Only the newest version aggregates; ``clip_bound`` (B)
    bounds each client's per-coordinate influence and fixes the quantization range
    (see shared/docs/secure-aggregation.md)."""

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
    """A client's seat in a round. The composite (round_id, user_id) primary key is
    the structural one-submission-per-client-per-round guard the protocol requires
    (a second masked vector under the same masks would leak the difference of two
    updates — reject it, don't merely rate-limit). ``ka_public_key`` is snapshotted
    at join, never joined to a mutable key table, so a client changing keys mid-round
    can't desync the roster the masks are derived against. ``masked`` is the client's
    submitted vector — m little-endian uint32 ring elements — set exactly once."""

    round_id: int = Field(foreign_key="secureround.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    ka_public_key: bytes       # 65-byte uncompressed P-256 point, snapshot at join
    masked: bytes | None = None
    submitted_at: datetime | None = None


class QuantizationJob(SQLModel, table=True):
    """Tracks one quantization request end to end. ``result`` holds the int8
    .tflite bytes and ``signature`` the server's ECDSA over its canonical model
    bytes (ml.payload); both are ephemeral — nulled by the cleanup sweep once
    served (after a grace period) or once expired (unclaimed)."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    submission_id: int = Field(foreign_key="clientdeltasubmission.id")
    model_key: str
    status: JobStatus = Field(default=JobStatus.pending)
    result: bytes | None = None
    signature: bytes | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    served_at: datetime | None = None


class Firmware(IntPKModel, table=True):
    """A published firmware build, seeded from a `shared/gen/firmware/{version}`
    export (`firmware/scripts/export_image.py`) and served by the /ota routes.
    The image itself lives on disk under SERVE_DIR, keyed by ``version``
    (common.storage); only its ``size`` (raw image bytes) and ``signature`` are
    kept here. ``interface_version`` is the BLE contract an app build must share
    to talk to it; ``supported_contracts`` are the model contract versions it
    can consume; ``signature`` (never null — an unverifiable image is useless to
    the device) is the server's ECDSA over the raw image bytes, verified by the
    device against its factory srv_pub before booting the image."""

    version: str = Field(unique=True, index=True)  # arbitrary <=32-byte build string
    interface_version: int = Field(index=True)
    supported_contracts: list[int] = Field(sa_column=Column(JSON))
    size: int                                      # raw (uncompressed) image size
    signature: bytes                               # DER ECDSA over the raw image
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
