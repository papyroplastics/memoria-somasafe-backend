"""Signed quantized-model payload served by /model/quantize/result and forwarded,
opaque, by the app to the ESP32. Layout (little-endian; server and ESP32 are both LE):

    u16   sig_len              byte length of the DER signature
    u8[]  signature            ECDSA P-256 (SHA-256) DER over the body below
    -- signed body --
    u16   payload_version      == PAYLOAD_VERSION
    u16   signature_version    == firmware-understood version; fixes the norm layout
    f32[] norm_params          z-score params, layout keyed by signature_version
    u8[]  tflite               the int8 model

The firmware verifies the signature against the factory-provisioned server public key,
checks both versions, applies the norm params (``(x - mean) / std``) and runs the tflite.
"""

import struct
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

PAYLOAD_VERSION = 1


def _load_private(key_path: Path) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{key_path}: expected an EC private key")
    return key


def build_payload(tflite: bytes, signature_version: int, norm_bytes: bytes,
                  key_path: Path) -> bytes:
    """Assemble and sign the payload wrapping ``tflite`` and its ``norm_bytes``."""
    body = (struct.pack('<HH', PAYLOAD_VERSION, signature_version)
            + norm_bytes + tflite)
    signature = _load_private(key_path).sign(body, ec.ECDSA(hashes.SHA256()))
    return struct.pack('<H', len(signature)) + signature + body


def verify_payload(payload: bytes, public_key: ec.EllipticCurvePublicKey) -> dict:
    """Verify and parse a payload (raises on a bad signature). Returns the parsed
    versions, the raw norm bytes and the tflite — used by tests and the firmware harness."""
    (sig_len,) = struct.unpack_from('<H', payload, 0)
    signature = payload[2:2 + sig_len]
    body = payload[2 + sig_len:]
    public_key.verify(signature, body, ec.ECDSA(hashes.SHA256()))
    payload_version, signature_version = struct.unpack_from('<HH', body, 0)
    rest = body[4:]
    return {
        'payload_version': payload_version,
        'signature_version': signature_version,
        'body': rest,   # norm_params ++ tflite; split by the caller per signature_version
    }
