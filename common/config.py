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
