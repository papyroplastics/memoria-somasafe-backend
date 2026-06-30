"""Seed the database with default rows: the model registry and a default user,
and optionally a device from a factory NVS partition definition.

Idempotent — run it after the services are up to bootstrap a fresh database:

    uv run -m scripts.db_seed                          # models + user
    uv run -m scripts.db_seed ../firmware/factory_nvs.csv # also seed that device
    uv run -m scripts.db_seed ../firmware/factory_nvs.csv --device-only
"""

import argparse
import csv
from pathlib import Path

from sqlmodel import Session, select

from api.routes.auth import hash_password
from common.config import RESULTS_DIR, SEED_EMAIL, SEED_PASSWORD, SEED_USER
from common.db import (
    Device,
    GlobalWeights,
    ModelDefinition,
    ModelFingerprint,
    User,
    engine,
    init_db,
)
from ml.model_list import MODELS
from ml.saving import load_trainable_weights


def seed_models(session: Session) -> None:
    """Seed each model that has a trained artifact on disk. Building the model
    yields its architecture fingerprint; the trainable .tflite seeds the initial
    GlobalWeights. Untrained models are skipped — rerun once they're trained."""
    for key, spec in MODELS.items():
        tflite = RESULTS_DIR / key / "trainable.tflite"
        if not tflite.exists():
            print(f"  - model '{key}' skipped (no {tflite})")
            continue

        trainer = MODELS[key].build_trainer()
        fingerprint = trainer.model.arch_fingerprint()
        param_count = trainer.model.total_parameter_size

        if session.get(ModelFingerprint, fingerprint) is None:
            session.add(ModelFingerprint(
                fingerprint=fingerprint,
                display_version=spec.version,
                param_count=param_count,
            ))

        meta = session.get(ModelDefinition, key)
        if meta is None:
            session.add(ModelDefinition(
                key=key, name=spec.name, purpose=spec.purpose,
                firmware_id=spec.firmware_id, app_version=spec.app_version,
                fingerprint=fingerprint,
            ))
            print(f"  + model '{key}' [{fingerprint}]")
        elif meta.fingerprint != fingerprint:
            meta.fingerprint = fingerprint
            session.add(meta)
            print(f"  ~ model '{key}' fingerprint -> {fingerprint}")

        has_weights = session.exec(
            select(GlobalWeights).where(
                GlobalWeights.model_key == key,
                GlobalWeights.fingerprint == fingerprint)
        ).first()
        if has_weights is None:
            params = load_trainable_weights(tflite)
            session.add(GlobalWeights(
                model_key=key, fingerprint=fingerprint,
                parameters=params.astype("float32").tobytes(),
                param_count=int(params.size),
            ))
            print(f"  + initial weights for '{key}' ({params.size} params)")
    session.commit()


def seed_users(session: Session) -> None:
    existing = session.exec(select(User).where(User.username == SEED_USER)).first()
    if existing is None:
        session.add(User(
            username=SEED_USER,
            email=SEED_EMAIL,
            hashed_password=hash_password(SEED_PASSWORD),
        ))
        session.commit()
        print(f"  + user '{SEED_USER}'")


def _parse_factory_nvs(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#") or row[0] == "key":
                continue
            if len(row) >= 4 and row[1] == "data":
                values[row[0]] = row[3]
    return values


def seed_device(session: Session, path: Path) -> None:
    fields = _parse_factory_nvs(path)
    serial, esp_pub = fields.get("serial"), fields.get("esp_pub")
    if not serial or not esp_pub:
        raise SystemExit(f"{path}: missing 'serial' or 'esp_pub'")

    if session.get(Device, serial) is None:
        session.add(Device(serial=serial, public_key=bytes.fromhex(esp_pub)))
        session.commit()
        print(f"  + device '{serial}'")
    else:
        print(f"  = device '{serial}' (already present)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory_nvs", type=Path, nargs="?",
                        help="factory NVS partition CSV to seed as a device")
    parser.add_argument("--device-only", action="store_true",
                        help="seed only the device, skipping the model registry and user")
    args = parser.parse_args()

    if args.device_only and args.factory_nvs is None:
        parser.error("--device-only requires a factory NVS CSV argument")

    init_db()
    with Session(engine) as session:
        if not args.device_only:
            seed_models(session)
            seed_users(session)
        if args.factory_nvs is not None:
            seed_device(session, args.factory_nvs)
    print("Seed complete.")


if __name__ == "__main__":
    main()
