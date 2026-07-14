import os
from pathlib import Path

# Time unit constants (all other time values in this file are in seconds)
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
WEEK = 7 * DAY

# Storage of the trained artifacts served as-is (the train.py outputs).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "shared/gen/models"))
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", "shared/gen/datasets"))

# MinIO/S3 bucket the gateway serves model/firmware/quantize-result blobs from,
# keyed by DB row so an object key is reconstructable from the row alone (see
# common/storage.py). The seed script populates it; aggregation appends to it.
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "localhost:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "somasafe")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "somasafe123")
S3_BUCKET = os.environ.get("S3_BUCKET", "somasafe")
S3_SECURE = os.environ.get("S3_SECURE", "false") == "true"

# ECDSA P-256 private key the worker signs quantized-model payloads with; its public
# half must be the srv_pub provisioned in the device's factory NVS (see shared/make_keys.sh
# and firmware/scripts/gen_factory_nvs.py).
SERVER_PRIVATE_KEY_FILE = Path(os.environ.get("SERVER_PRIVATE_KEY", "shared/gen/server-private-key.pem"))

# PostgreSQL (SQLModel/SQLAlchemy URL) and Redis instance.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://somasafe:somasafe@localhost:5432/somasafe")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Quantization-result lifetime. A served result is kept for SERVE_GRACE_SECONDS
# so the client can retry the download; an unclaimed one is kept up to
# RESULT_TTL_SECONDS. The cleanup sweep runs every CLEANUP_INTERVAL_SECONDS.
SERVE_GRACE_SECONDS = int(os.environ.get("SERVE_GRACE_SECONDS", MINUTE * 5))
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", HOUR))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", MINUTE * 2))
# How long the quantize-result endpoint blocks on the task before returning 202
# so the client re-polls (the job id is the Celery task id it waits on).
RESULT_POLL_TIMEOUT_SECONDS = int(os.environ.get("RESULT_POLL_TIMEOUT_SECONDS", 30))

# Seed used to rebuild the representative dataset for int8 calibration.
SEED = int(os.environ.get("SEED", 1234))

# --- Federated aggregation (see worker.tasks.federated_aggregation) ---
FED_AGG_INTERVAL_SECONDS = int(os.environ.get("FED_AGG_INTERVAL_SECONDS", DAY))
# Minimum valid submissions a model needs in the window for a round to run.
FED_MIN_SUBMISSIONS = int(os.environ.get("FED_MIN_SUBMISSIONS", 1))

# --- Secure aggregation (see worker.tasks.secure_aggregation) ---
# Per-coordinate clipping bound B: each client clips its delta to +/-B before
# masking, capping its influence on the mean to B/n. Also fixes the fixed-point
# range, so it must comfortably exceed real delta magnitudes (a generous default
# — with ~15 clients there is ample headroom before the ring can wrap).
SECURE_CLIP_BOUND = float(os.environ.get("SECURE_CLIP_BOUND", 1.0))
# A round must have at least this many members to seal (n >= 3: the sum of two
# updates plus one own value reveals the third).
SECURE_MIN_MEMBERS = int(os.environ.get("SECURE_MIN_MEMBERS", 3))

# --- Auth (stateful opaque tokens: access in Redis, refresh in Postgres — see
# api.lib.session and api.routes.auth) ---
ACCESS_TOKEN_TTL_SECONDS = int(os.environ.get("ACCESS_TOKEN_TTL_SECONDS", MINUTE * 30))
REFRESH_TOKEN_TTL_SECONDS = int(os.environ.get("REFRESH_TOKEN_TTL_SECONDS", DAY * 30))

# Default account created by scripts.seed (no public registration).
SEED_USER = os.environ.get("SEED_USER", "somasafe")
SEED_PASSWORD = os.environ.get("SEED_PASSWORD", "somasafe")
SEED_EMAIL = os.environ.get("SEED_EMAIL") or None

# --- Rate limiting ---
# Per-user, per-model cooldown between artifact downloads (trainable/quantized).
DOWNLOAD_COOLDOWN_SECONDS = int(os.environ.get("DOWNLOAD_COOLDOWN_SECONDS", MINUTE * 5))
# Per-user, per-interface cooldown between firmware image downloads.
OTA_DOWNLOAD_COOLDOWN_SECONDS = int(os.environ.get("OTA_DOWNLOAD_COOLDOWN_SECONDS", MINUTE * 5))
# Per-user, per-model daily cap on quantization submissions.
QUANTIZE_DAILY_LIMIT = int(os.environ.get("QUANTIZE_DAILY_LIMIT", 2))
QUANTIZE_DAILY_WINDOW_SECONDS = int(os.environ.get("QUANTIZE_DAILY_WINDOW_SECONDS", DAY))
# Per-user, per-model daily cap on submit-only weight uploads.
SUBMIT_DAILY_LIMIT = int(os.environ.get("SUBMIT_DAILY_LIMIT", 2))
SUBMIT_DAILY_WINDOW_SECONDS = int(os.environ.get("SUBMIT_DAILY_WINDOW_SECONDS", DAY))

# --- Device attestation (see api.routes.device) ---
# How long an issued ownership challenge stays valid before it must be reissued.
DEVICE_CHALLENGE_TTL_SECONDS = int(os.environ.get("DEVICE_CHALLENGE_TTL_SECONDS", MINUTE * 5))
# A device's owner may only change once per this window (24 h since the last
# successful attestation). Failed/timed-out challenges do not count.
DEVICE_ATTEST_COOLDOWN_SECONDS = int(os.environ.get("DEVICE_ATTEST_COOLDOWN_SECONDS", DAY))
