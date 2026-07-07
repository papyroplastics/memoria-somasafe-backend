"""Server signature over a distributed model, verified by the ESP32 against its
factory-provisioned server public key before loading.

The signature is transport-independent: the app packages the model for the
device however its BLE interface version dictates, and the firmware rebuilds
the canonical byte string below from the delivered fields and verifies the
signature over it (see shared/docs/model-signing.md). Layout (little-endian):

    u16   contract_version     fixes how the model is fed: the norm layout + I/O signatures
    f32[] norm_params          z-score params; count is fixed by contract_version
    u8[]  tflite               the int8 model

The gateway delivers the three fields alongside the tflite in response headers
(X-Model-Signature / X-Contract-Version / X-Norm-Params, base64 where binary).
The device applies the norm params as ``(x - mean) / std`` before the model.
"""

import struct
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _load_private(key_path: Path) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{key_path}: expected an EC private key")
    return key


def canonical_model_bytes(tflite: bytes, contract_version: int, norm_params: bytes) -> bytes:
    return struct.pack('<H', contract_version) + norm_params + tflite


def sign_blob(data: bytes, key_path: Path) -> bytes:
    """ECDSA P-256 (SHA-256) DER signature over raw bytes (e.g. a firmware image)."""
    return _load_private(key_path).sign(data, ec.ECDSA(hashes.SHA256()))


def sign_model(tflite: bytes, contract_version: int, norm_params: bytes,
               key_path: Path) -> bytes:
    """ECDSA P-256 (SHA-256) DER signature over the canonical model bytes."""
    return sign_blob(canonical_model_bytes(tflite, contract_version, norm_params),
                     key_path)


def verify_model(signature: bytes, tflite: bytes, contract_version: int,
                 norm_params: bytes, public_key: ec.EllipticCurvePublicKey) -> None:
    """Raises InvalidSignature on mismatch — used by tests and the firmware harness."""
    public_key.verify(signature,
                      canonical_model_bytes(tflite, contract_version, norm_params),
                      ec.ECDSA(hashes.SHA256()))
