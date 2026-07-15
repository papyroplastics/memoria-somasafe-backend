"""Seed the database with default rows: the model registry, exported firmware
builds, a default user, and a device from a factory NVS partition definition.

Idempotent — run it after the services are up to bootstrap a fresh database:

    uv run -m scripts.system.seed_db # use default nvs path
    uv run -m scripts.system.seed_db <nvs definition csv> # use specific device
    uv run -m scripts.system.seed_db --assign-device # assign the device to the seed user
"""

import argparse
import csv
import json
from datetime import datetime, timezone
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
from common.compression import compress
from common.db import (
    Artifact,
    ClientDeltaSubmission,
    Device,
    Firmware,
    FirmwareImage,
    GlobalWeights,
    ModelDefinition,
    ModelVersion,
    QuantizationJob,
    SecureRound,
    SecureRoundMember,
    User,
    WeightsArtifact,
    engine,
    get_latest_version,
    init_db,
    utcnow,
)
from ml.preprocessing import CLEAN_SUBDIR, get_sorted_paths
from ml.model_list import MODELS
from ml.payload import sign_blob, sign_model
from ml.saving import load_trainable_weights

default_nvs = "shared/gen/factory_nvs.csv"
default_firmware_dir = "shared/gen/firmware"

def reset_weights(session: Session, key: str) -> None:
    weights = session.exec(
        select(GlobalWeights).where(GlobalWeights.model_key == key)).all()
    if not weights:
        return
    weights_ids = [w.id for w in weights]

    submissions = session.exec(
        select(ClientDeltaSubmission)
        .where(ClientDeltaSubmission.base_weights_id.in_(weights_ids))  # type: ignore[attr-defined]
    ).all()
    jobs = session.exec(
        select(QuantizationJob)
        .where(QuantizationJob.submission_id.in_([s.id for s in submissions]))  # type: ignore[attr-defined]
    ).all() if submissions else []
    rounds = session.exec(
        select(SecureRound)
        .where(SecureRound.base_weights_id.in_(weights_ids))  # type: ignore[attr-defined]
    ).all()
    members = session.exec(
        select(SecureRoundMember)
        .where(SecureRoundMember.round_id.in_([r.id for r in rounds]))  # type: ignore[attr-defined]
    ).all() if rounds else []

    # Children first: each level is a foreign key into the next. Their blob rows
    # (QuantizationResult, WeightsArtifact) cascade at the DB level.
    for job in jobs:
        session.delete(job)
    for submission in submissions:
        session.delete(submission)
    for member in members:
        session.delete(member)
    for round_ in rounds:
        session.delete(round_)
    for w in weights:
        session.delete(w)
    session.flush()

    print(f"  - reset '{key}': dropped {len(weights)} weight snapshot(s), "
          f"{len(submissions)} submission(s), {len(jobs)} quantization job(s), "
          f"{len(rounds)} secure round(s)")


def seed_models(session: Session, reset: bool = False) -> None:
    for key, spec in MODELS.items():
        tflite = MODELS_DIR / key / "trainable.tflite"
        if not tflite.exists():
            print(f"  - model '{key}' skipped (no {tflite})")
            continue

        trainer = spec.build_trainer(DATASETS_DIR)
        fingerprint = trainer.arch_fingerprint()

        if session.get(ModelDefinition, key) is None:
            session.add(ModelDefinition(
                key=key, name=spec.name,
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
                weight_count=trainer.model.total_weight_size,
                contract_version=trainer.contract_version,
                submission_type=spec.submission_type,
                norm_params=trainer.norm_param_bytes(),
                min_app_version=spec.min_app_version,
            )
            session.add(version)
            session.flush()
            print(f"  + model '{key}' v{spec.version} [{fingerprint}]")

        if reset:
            reset_weights(session, key)

        has_weights = session.exec(
            select(GlobalWeights).where(GlobalWeights.version_id == version.id)
        ).first()
        if has_weights is None:
            weights = load_trainable_weights(tflite)
            trainable_bytes = tflite.read_bytes()
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
            gw = GlobalWeights(
                model_key=key, version_id=version.id,
                weights=weights.astype("float32").tobytes(),
                weight_count=int(weights.size),
                artifact_signature=signature,
            )
            session.add(gw)
            session.flush()  # need the row id the artifacts are keyed by
            session.add(WeightsArtifact(weights_id=gw.id,
                                        artifact=Artifact.trainable,
                                        data=compress(trainable_bytes)))
            if quantized is not None:
                session.add(WeightsArtifact(weights_id=gw.id,
                                            artifact=Artifact.quantized,
                                            data=compress(quantized)))
            print(f"  + initial weights for '{key}' v{spec.version} ({weights.size} weights)")
    session.commit()


def seed_firmware(session: Session, firmware_dir: Path) -> None:
    if not firmware_dir.is_dir():
        print(f"  - firmware skipped (no {firmware_dir})")
        return

    for entry in sorted(firmware_dir.iterdir()):
        metadata_file = entry / "metadata.json"
        image_file = entry / "firmware.bin"
        if not metadata_file.exists() or not image_file.exists():
            continue

        metadata = json.loads(metadata_file.read_text())
        version = metadata["version"]
        if session.exec(select(Firmware).where(Firmware.version == version)).first():
            print(f"  = firmware '{version}' (already present)")
            continue

        if not SERVER_PRIVATE_KEY_FILE.exists():
            print(f"  ! no key at {SERVER_PRIVATE_KEY_FILE}; "
                  f"firmware '{version}' skipped (signature required)")
            continue

        blob = image_file.read_bytes()
        signature = sign_blob(blob, SERVER_PRIVATE_KEY_FILE)

        created_at = datetime.fromisoformat(metadata["created_at"])
        if created_at.tzinfo is not None:
            created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
        firmware = Firmware(
            version=version,
            interface_version=metadata["interface_version"],
            supported_contracts=metadata["supported_contracts"],
            size=len(blob),
            signature=signature,
            created_at=created_at,
        )
        session.add(firmware)
        session.flush()  # need the row id the image is keyed by
        session.add(FirmwareImage(firmware_id=firmware.id, data=compress(blob)))
        print(f"  + firmware '{version}' (interface {metadata['interface_version']}, "
              f"contracts {metadata['supported_contracts']}, {len(blob)} bytes)")
    session.commit()


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


def seed_test_users(session: Session) -> None:
    subjects = get_sorted_paths(DATASETS_DIR / CLEAN_SUBDIR)
    if not subjects:
        raise SystemExit(f"no subjects under {DATASETS_DIR / CLEAN_SUBDIR}; "
                         f"run scripts/get_dataset.py first")

    # Attestation is bypassed for these fakes, so the stored key is never used;
    # a well-formed 65-byte uncompressed P-256 point is enough.
    placeholder_pubkey = b"\x04" + bytes(64)
    for i in range(1, len(subjects) + 1):
        name = f"test_{i}"
        user = session.exec(select(User).where(User.username == name)).first()
        if user is None:
            user = User(username=name, hashed_password=hash_password(name))
            session.add(user)
            session.commit()
            session.refresh(user)

        serial = f"TEST-DEVICE-{i}"
        device = session.get(Device, serial)
        if device is None:
            device = Device(serial=serial, public_key=placeholder_pubkey)
            session.add(device)
        device.owner_id = user.id
        device.last_attested_at = utcnow()
    session.commit()
    print(f"  + {len(subjects)} test users with owned devices")


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
    parser.add_argument("--firmware-dir", type=Path, default=Path(default_firmware_dir),
                        help="directory of exported firmware versions to seed")
    parser.add_argument("--assign-device", action="store_true",
                        help="assign the seeded device to the seed user, even if either already existed")
    parser.add_argument("--test-users", action="store_true",
                        help="create a test_N user (owning a placeholder device) per "
                             "dataset subject, for the headless federated harness")
    parser.add_argument("--reset-weights", action="store_true",
                        help="drop each seeded model's weight snapshots (and the "
                             "submissions, quantization jobs and secure rounds based on "
                             "them) and re-seed from the artifacts now on disk — use "
                             "after retraining a model that is already seeded")
    args = parser.parse_args()

    if not args.factory_nvs.exists():
        parser.error(f"{args.factory_nvs} does not exist.")

    init_db()
    with Session(engine) as session:
        seed_models(session, reset=args.reset_weights)
        seed_firmware(session, args.firmware_dir)
        user = seed_users(session)
        seed_device(session, args.factory_nvs, user if args.assign_device else None)
        if args.test_users:
            seed_test_users(session)
    print("Seed complete.")


if __name__ == "__main__":
    main()
