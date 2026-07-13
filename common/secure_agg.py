"""Primitives for minimal secure aggregation (honest-but-curious server), shared
by the client harness, the worker, and the tests. TensorFlow-free and pure: the
masking is client-side and platform-independent, the summation is integer work.

The scheme (shared/docs/secure-aggregation.md): every client holds a long-term
ECDH keypair; for each pair (u, v) both sides independently derive the same PRG
seed from their shared secret and the round id, expand it to a mask, and the
lower-indexed member adds the mask while the higher subtracts it. Summed over the
whole cohort every pairwise mask cancels, leaving the plain sum of the quantized
updates and nothing else.

All ring arithmetic is mod 2^32; quantization maps clipped float deltas into that
ring so the masked sum is exact (floats don't cancel).
"""

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

RING_MODULUS = 2**32
_HALF = 2**31
_INFO = b"secagg-v1"

Roster = list[tuple[int, bytes]]  # (user_id, uncompressed P-256 public point)


def generate_keypair() -> tuple[ec.EllipticCurvePrivateKey, bytes]:
    sk = ec.generate_private_key(ec.SECP256R1())
    pk = sk.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    return sk, pk


def derive_seed(sk: ec.EllipticCurvePrivateKey, pk_other: bytes,
                round_id: int) -> bytes:
    peer = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pk_other)
    shared = sk.exchange(ec.ECDH(), peer)
    return HKDF(algorithm=hashes.SHA256(), length=32,
                salt=round_id.to_bytes(8, "big"), info=_INFO).derive(shared)


def prg(seed: bytes, m: int) -> np.ndarray:
    encryptor = Cipher(algorithms.AES(seed), modes.CTR(b"\x00" * 16)).encryptor()
    keystream = encryptor.update(b"\x00" * (4 * m)) + encryptor.finalize()
    return np.frombuffer(keystream, dtype="<u4").astype(np.uint64)


def compute_scale(n: int, clip_bound: float) -> int:
    return int(_HALF // (n * clip_bound))


def mask_vector(q: np.ndarray, self_id: int, roster: Roster,
                sk: ec.EllipticCurvePrivateKey, round_id: int) -> np.ndarray:
    y = q.astype(np.uint64)
    for uid, pk_other in roster:
        if uid == self_id:
            continue
        mask = prg(derive_seed(sk, pk_other, round_id), q.size)
        # Modular add only (uint64 subtraction underflows), keeping y < 2^32.
        signed = mask if self_id < uid else (RING_MODULUS - mask)
        y = (y + signed) % RING_MODULUS
    return y.astype(np.uint32)


def ring_sum(vectors: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(vectors[0].size, dtype=np.uint64)
    for v in vectors:
        acc = (acc + v.astype(np.uint64)) % RING_MODULUS
    return acc.astype(np.uint32)


def quantize(delta: np.ndarray, clip_bound: float, scale: int) -> np.ndarray:
    clipped = np.clip(delta.astype(np.float64), -clip_bound, clip_bound)
    return (np.rint(clipped * scale).astype(np.int64) % RING_MODULUS).astype(np.uint32)

def dequantize(z: np.ndarray, scale: int, n: int) -> np.ndarray:
    signed = z.astype(np.int64)
    signed = np.where(signed >= _HALF, signed - RING_MODULUS, signed)
    return (signed / scale / n).astype(np.float32)
