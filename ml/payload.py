"""Server signature over a distributed model, verified by the ESP32 against its
factory-provisioned public key before loading (see shared/docs/model-signing.md).
Signs the contract version + tflite bytes; per-wearer z-score params are not covered."""

import struct
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _load_private(key_path: Path) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{key_path}: expected an EC private key")
    return key


def canonical_model_bytes(tflite: bytes, contract_version: int) -> bytes:
    return struct.pack('<H', contract_version) + tflite


def sign_blob(data: bytes, key_path: Path) -> bytes:
    """ECDSA P-256 (SHA-256) DER signature over raw bytes (e.g. a firmware image)."""
    return _load_private(key_path).sign(data, ec.ECDSA(hashes.SHA256()))


def sign_model(tflite: bytes, contract_version: int, key_path: Path) -> bytes:
    """ECDSA P-256 (SHA-256) DER signature over the canonical model bytes."""
    return sign_blob(canonical_model_bytes(tflite, contract_version), key_path)


def verify_model(signature: bytes, tflite: bytes, contract_version: int,
                 public_key: ec.EllipticCurvePublicKey) -> None:
    """Raises InvalidSignature on mismatch."""
    public_key.verify(signature, canonical_model_bytes(tflite, contract_version),
                      ec.ECDSA(hashes.SHA256()))
