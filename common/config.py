import os
from pathlib import Path

# Storage of the trained artifacts served as-is (the train.py outputs).
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", "datasets"))

# PostgreSQL (SQLModel/SQLAlchemy URL) and the Celery broker.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://somasafe:somasafe@localhost:5432/somasafe")
BROKER_URL = os.environ.get("BROKER_URL", "redis://localhost:6379/0")

# Quantization-result lifetime. A served result is kept for SERVE_GRACE_SECONDS
# so the client can retry the download; an unclaimed one is kept up to
# RESULT_TTL_SECONDS. The cleanup sweep runs every CLEANUP_INTERVAL_SECONDS.
SERVE_GRACE_SECONDS = int(os.environ.get("SERVE_GRACE_SECONDS", 300))
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", 3600))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", 120))

# Seed used to rebuild the representative dataset for int8 calibration.
SEED = int(os.environ.get("SEED", 1234))

# --- Auth (stateful opaque tokens, see api.auth) ---
ACCESS_TOKEN_TTL_SECONDS = int(os.environ.get("ACCESS_TOKEN_TTL_SECONDS", 1800))      # 30 min
REFRESH_TOKEN_TTL_SECONDS = int(os.environ.get("REFRESH_TOKEN_TTL_SECONDS", 2592000))  # 30 days

# Default account created by scripts.seed (no public registration).
SEED_USER = os.environ.get("SEED_USER", "somasafe")
SEED_PASSWORD = os.environ.get("SEED_PASSWORD", "somasafe")
SEED_EMAIL = os.environ.get("SEED_EMAIL") or None

# --- Rate limiting (Redis-backed, see common.ratelimit) ---
# Separate Redis db from the Celery broker (db 0) to keep counters isolated.
RATELIMIT_URL = os.environ.get("RATELIMIT_URL", "redis://localhost:6379/1")
# Per-user, per-model cooldown between artifact downloads (trainable/quantized).
DOWNLOAD_COOLDOWN_SECONDS = int(os.environ.get("DOWNLOAD_COOLDOWN_SECONDS", 300))
# Per-user, per-model daily cap on quantization submissions.
QUANTIZE_DAILY_LIMIT = int(os.environ.get("QUANTIZE_DAILY_LIMIT", 2))
QUANTIZE_DAILY_WINDOW_SECONDS = int(os.environ.get("QUANTIZE_DAILY_WINDOW_SECONDS", 86400))

# --- Device attestation (see api.device) ---
# How long an issued ownership challenge stays valid before it must be reissued.
DEVICE_CHALLENGE_TTL_SECONDS = int(os.environ.get("DEVICE_CHALLENGE_TTL_SECONDS", 300))
# A device's owner may only change once per this window (24 h since the last
# successful attestation). Failed/timed-out challenges do not count.
DEVICE_ATTEST_COOLDOWN_SECONDS = int(os.environ.get("DEVICE_ATTEST_COOLDOWN_SECONDS", 86400))
