import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name} (see example.env)")
    return value


# Time unit constants (all other time values in this file are in seconds)
MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
WEEK = 7 * DAY

# Storage of the trained artifacts served as-is (the train.py outputs).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "shared/gen/models"))
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", "shared/gen/datasets"))
CALIBRATION_DIR = Path(os.environ.get("CALIBRATION_DIR", "shared/gen/calibration"))
# Subject exports (the .ssds capture protobuf) the app imports and the firmware harness
# streams; both read them straight out of shared/gen.
EXPORTS_DIR = Path(os.environ.get("EXPORTS_DIR", "shared/gen/exports"))

# Evaluation/experiment outputs (histories, reports, figures, distilled labels).
# Separate from MODELS_DIR: those are serving artifacts the system consumes, this
# is everything the thesis measures. Gitignored.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

# ECDSA P-256 private key the worker signs quantized-model payloads with; its public
# half must be the srv_pub provisioned in the device's factory NVS (see shared/Makefile
# and firmware/scripts/gen_factory_nvs.py).
SERVER_PRIVATE_KEY_FILE = Path(os.environ.get("SERVER_PRIVATE_KEY", "shared/gen/server-private-key.pem"))

# PostgreSQL (SQLModel/SQLAlchemy URL) and Redis instance.
DATABASE_URL = _require("DATABASE_URL")
REDIS_URL = _require("REDIS_URL")

# Celery broker/result-backend Redis, separate from REDIS_URL (auth sessions and
# rate limiting). Defaults to REDIS_URL so a single-instance local setup still
# works unchanged.
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

# Quantization-result lifetime. A served result is kept for SERVE_GRACE_SECONDS
# so the client can retry the download; an unclaimed one is kept up to
# RESULT_TTL_SECONDS. The cleanup sweep runs every CLEANUP_INTERVAL_SECONDS.
SERVE_GRACE_SECONDS = int(os.environ.get("SERVE_GRACE_SECONDS", MINUTE * 5))
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", HOUR))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", MINUTE * 2))
CLEANUP_BATCH_SIZE = int(os.environ.get("CLEANUP_BATCH_SIZE", 500))

# --- Worker (see worker.celery_app) ---
# Must hold soft < hard < FED_LOCK_TTL_SECONDS < CELERY_VISIBILITY_TIMEOUT_SECONDS.
WORKER_TASK_SOFT_TIME_LIMIT = int(os.environ.get("WORKER_TASK_SOFT_TIME_LIMIT", MINUTE * 5))
WORKER_TASK_TIME_LIMIT = int(os.environ.get("WORKER_TASK_TIME_LIMIT", MINUTE * 6))
CELERY_VISIBILITY_TIMEOUT_SECONDS = int(os.environ.get("CELERY_VISIBILITY_TIMEOUT_SECONDS", HOUR))
# A claimed row older than this was left by a dead worker and is reaped.
WORKER_REAP_AFTER_SECONDS = WORKER_TASK_TIME_LIMIT + MINUTE

# RNG seed used globally
SEED = int(os.environ.get("SEED", 1234))

# --- Federated aggregation (see worker.tasks.aggregation) ---
FED_AGG_INTERVAL_SECONDS = int(os.environ.get("FED_AGG_INTERVAL_SECONDS", DAY))
FED_LOCK_TTL_SECONDS = int(os.environ.get("FED_LOCK_TTL_SECONDS", MINUTE * 7))
# Memory budget for a round's stacked deltas; caps the cohort at the newest
# FED_AGG_MEMORY_BYTES // (weight_count * 4) submissions.
FED_AGG_MEMORY_BYTES = int(os.environ.get("FED_AGG_MEMORY_BYTES", 500 * 1024 * 1024))
# Minimum valid submissions a model needs in the window for a round to run.
FED_MIN_SUBMISSIONS = int(os.environ.get("FED_MIN_SUBMISSIONS", 1))
# Fraction of values the trimmed-mean aggregator drops from each side of every
# coordinate. Must be in [0, 0.5); below 1/n it trims nothing and is a plain mean.
FED_TRIM_RATIO = float(os.environ.get("FED_TRIM_RATIO", 0.2))

# --- Secure aggregation (see worker.tasks.secure) ---
# Per-coordinate clipping bound B: each client clips its delta to +/-B before
# masking, capping its influence on the mean to B/n. Also fixes the fixed-point
# range, so it must comfortably exceed real delta magnitudes (a generous default
# — with ~15 clients there is ample headroom before the ring can wrap).
SECURE_CLIP_BOUND = float(os.environ.get("SECURE_CLIP_BOUND", 1.0))
# A round must have at least this many members to seal (n >= 3: the sum of two
# updates plus one own value reveals the third).
SECURE_MIN_MEMBERS = int(os.environ.get("SECURE_MIN_MEMBERS", 3))
# The sweep seals an open round at SECURE_TARGET_MEMBERS, or once it has been open
# for SECURE_ROUND_OPEN_TIMEOUT_SECONDS with at least SECURE_MIN_MEMBERS; a sealed
# round missing submissions after SECURE_ROUND_SEAL_TIMEOUT_SECONDS fails.
SECURE_TARGET_MEMBERS = int(os.environ.get("SECURE_TARGET_MEMBERS", 10))
SECURE_ROUND_OPEN_TIMEOUT_SECONDS = int(os.environ.get("SECURE_ROUND_OPEN_TIMEOUT_SECONDS", MINUTE * 30))
SECURE_ROUND_SEAL_TIMEOUT_SECONDS = int(os.environ.get("SECURE_ROUND_SEAL_TIMEOUT_SECONDS", MINUTE * 30))
SECURE_SWEEP_INTERVAL_SECONDS = int(os.environ.get("SECURE_SWEEP_INTERVAL_SECONDS", 30))

# --- Auth (stateful opaque tokens: access in Redis, refresh in Postgres — see
# api.lib.session and api.routes.auth) ---
ACCESS_TOKEN_TTL_SECONDS = int(os.environ.get("ACCESS_TOKEN_TTL_SECONDS", MINUTE * 30))
REFRESH_TOKEN_TTL_SECONDS = int(os.environ.get("REFRESH_TOKEN_TTL_SECONDS", DAY * 30))

# Default account created by scripts.seed (no public registration).
SEED_USER = _require("SEED_USER")
SEED_PASSWORD = _require("SEED_PASSWORD")
SEED_EMAIL = os.environ.get("SEED_EMAIL") or None

# --- Rate limiting ---
# Per-user, per-model cooldown between artifact downloads (trainable/quantized).
DOWNLOAD_COOLDOWN_SECONDS = int(os.environ.get("DOWNLOAD_COOLDOWN_SECONDS", MINUTE * 5))
# Per-user, per-interface cooldown between firmware image downloads.
OTA_DOWNLOAD_COOLDOWN_SECONDS = int(os.environ.get("OTA_DOWNLOAD_COOLDOWN_SECONDS", MINUTE * 5))
# Per-user, per-model daily cap on weight submissions, shared by raw, quantize
# and secure submit paths (they all cost the same budget).
SUBMIT_DAILY_LIMIT = int(os.environ.get("SUBMIT_DAILY_LIMIT", 2))
SUBMIT_DAILY_WINDOW_SECONDS = int(os.environ.get("SUBMIT_DAILY_WINDOW_SECONDS", DAY))
# Per-user, per-model cooldown between secure-round joins.
SECURE_JOIN_COOLDOWN_SECONDS = int(os.environ.get("SECURE_JOIN_COOLDOWN_SECONDS", MINUTE * 5))

# --- Device attestation (see api.routes.device) ---
# How long an issued ownership challenge stays valid before it must be reissued.
DEVICE_CHALLENGE_TTL_SECONDS = int(os.environ.get("DEVICE_CHALLENGE_TTL_SECONDS", MINUTE * 5))
# A device's owner may only change once per this window (24 h since the last
# successful attestation). Failed/timed-out challenges do not count.
DEVICE_ATTEST_COOLDOWN_SECONDS = int(os.environ.get("DEVICE_ATTEST_COOLDOWN_SECONDS", DAY))

# Disable tqdm globally
DISABLE_TQDM = os.getenv("DISABLE_TQDM", "").lower() in ("1", "true", "yes")

