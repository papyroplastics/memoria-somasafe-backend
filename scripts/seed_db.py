"""Seed the database with default rows: the model registry, a default user,
and a device from a factory NVS partition definition.

Idempotent — run it after the services are up to bootstrap a fresh database:

    uv run -m scripts.seed_db # use default nvs path
    uv run -m scripts.seed_db <nvs definition csv> # use specific device
    uv run -m scripts.seed_db --assign-device # assign the device to the seed user
"""

import argparse
import csv
from pathlib import Path

from sqlmodel import Session, select

from api.routes.auth import hash_password
from common.config import (
    DATASETS_DIR,
    MODELS_DIR,
    SEED_EMAIL,
    SEED_PASSWORD,
    SEED_USER,
    SERVER_PRIVATE_KEY_FILE,
)
from common.db import (
    Device,
    Firmware,
    GlobalWeights,
    ModelDefinition,
    ModelVersion,
    User,
    engine,
    get_latest_version,
    init_db,
    utcnow,
)
from ml.model_list import MODELS
from ml.payload import sign_model
from ml.saving import load_trainable_weights

default_nvs = "shared/gen/factory_nvs.csv"

def seed_models(session: Session) -> None:
    """Seed each model that has a trained artifact on disk. Building the trainer
    yields the fingerprint the registry version is checked against (a moved
    fingerprint without a ModelSpec.version bump aborts the seed); the trained
    artifacts seed the version's initial GlobalWeights. Untrained models are
    skipped — rerun once they're trained."""
    for key, spec in MODELS.items():
        tflite = MODELS_DIR / key / "trainable.tflite"
        if not tflite.exists():
            print(f"  - model '{key}' skipped (no {tflite})")
            continue

        trainer = spec.build_trainer(DATASETS_DIR)
        fingerprint = trainer.arch_fingerprint()

        if session.get(ModelDefinition, key) is None:
            session.add(ModelDefinition(
                key=key, name=spec.name, purpose=spec.purpose,
                firmware_id=spec.firmware_id,
            ))
            print(f"  + model '{key}'")

        latest = get_latest_version(session, key)
        if latest is not None and spec.version < latest.version:
            raise SystemExit(
                f"model '{key}': registry version {spec.version} is older than the "
                f"seeded v{latest.version}")
        if latest is not None and spec.version == latest.version:
            if latest.fingerprint != fingerprint:
                raise SystemExit(
                    f"model '{key}': fingerprint changed ({latest.fingerprint} -> "
                    f"{fingerprint}) but the registry still says v{spec.version} — "
                    f"bump ModelSpec.version")
            version = latest
        else:
            version = ModelVersion(
                model_key=key, version=spec.version, fingerprint=fingerprint,
                param_count=trainer.model.total_parameter_size,
                contract_version=trainer.contract_version,
                norm_params=trainer.norm_param_bytes(),
                min_app_version=spec.min_app_version,
            )
            session.add(version)
            session.flush()
            print(f"  + model '{key}' v{spec.version} [{fingerprint}]")

        has_weights = session.exec(
            select(GlobalWeights).where(GlobalWeights.version_id == version.id)
        ).first()
        if has_weights is None:
            params = load_trainable_weights(tflite)
            quantized_file = MODELS_DIR / key / "quantized.tflite"
            quantized = quantized_file.read_bytes() if quantized_file.exists() else None
            signature = None
            if quantized is not None:
                if SERVER_PRIVATE_KEY_FILE.exists():
                    signature = sign_model(quantized, version.contract_version,
                                           version.norm_params, SERVER_PRIVATE_KEY_FILE)
                else:
                    print(f"  ! no key at {SERVER_PRIVATE_KEY_FILE}; "
                          f"'{key}' quantized artifact left unsigned")
            session.add(GlobalWeights(
                model_key=key, version_id=version.id,
                parameters=params.astype("float32").tobytes(),
                param_count=int(params.size),
                trainable_artifact=tflite.read_bytes(),
                quantized_artifact=quantized,
                artifact_signature=signature,
            ))
            print(f"  + initial weights for '{key}' v{spec.version} ({params.size} params)")
    session.commit()


def seed_firmware(session: Session) -> None:
    if session.exec(select(Firmware)).first() is None:
        session.add(Firmware(version="1.0.0", interface_version=1,
                             supported_contract_version=1))
        session.commit()
        print("  + firmware '1.0.0' (interface 1, contract 1)")


def seed_users(session: Session) -> User:
    user = session.exec(select(User).where(User.username == SEED_USER)).first()
    if user is None:
        user = User(
            username=SEED_USER,
            email=SEED_EMAIL,
            hashed_password=hash_password(SEED_PASSWORD),
        )
        session.add(user)
        session.commit()
        print(f"  + user '{SEED_USER}'")
    return user


def _parse_factory_nvs(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#") or row[0] == "key":
                continue
            if len(row) >= 4 and row[1] == "data":
                values[row[0]] = row[3]
    return values


def seed_device(session: Session, path: Path, user: User | None) -> None:
    fields = _parse_factory_nvs(path)
    serial, esp_pub = fields.get("serial"), fields.get("esp_pub")
    if not serial or not esp_pub:
        raise SystemExit(f"{path}: missing 'serial' or 'esp_pub'")

    device = session.get(Device, serial)
    if device is None:
        device = Device(serial=serial, public_key=bytes.fromhex(esp_pub))
        session.add(device)
        print(f"  + device '{serial}'")
    else:
        print(f"  = device '{serial}' (already present)")

    if user is not None:
        device.owner_id = user.id
        device.last_attested_at = utcnow()
        print(f"  + assigned device '{serial}' to user '{user.username}'")

    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory_nvs", nargs='?', type=Path, default=Path(default_nvs),
                        help="factory NVS partition CSV to seed as a device")
    parser.add_argument("--assign-device", action="store_true",
                        help="assign the seeded device to the seed user, even if either already existed")
    args = parser.parse_args()

    if not args.factory_nvs.exists():
        parser.error(f"{args.factory_nvs} does not exist.")

    init_db()
    with Session(engine) as session:
        seed_models(session)
        seed_firmware(session)
        user = seed_users(session)
        seed_device(session, args.factory_nvs, user if args.assign_device else None)
    print("Seed complete.")


if __name__ == "__main__":
    main()
