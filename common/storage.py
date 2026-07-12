"""On-disk store for the blobs the gateway serves (model artifacts, firmware
images). Files live under SERVE_DIR, zstd-compressed, at paths derived purely
from the DB row that owns them — so a handler can locate a file from the row it
already loaded, with no filename column. This assumes the tree and the database
stay in sync; the seed script rebuilds it from scratch.

Bytes are stored and served compressed: the client decompresses. Signatures in
the DB cover the *raw* bytes, so compression happens after signing and the
client verifies after decompressing.
"""

from pathlib import Path

import zstandard

from common.config import SERVE_DIR

_compressor = zstandard.ZstdCompressor(level=19)
_decompressor = zstandard.ZstdDecompressor()
_ext = "zst"

def compress(data: bytes) -> bytes:
    return _compressor.compress(data)


def decompress(data: bytes) -> bytes:
    return _decompressor.decompress(data)


def weights_artifact_path(model_key: str, version_id: int, weights_id: int,
                          artifact: str) -> Path:
    """File for one serving artifact (``artifact`` is ``trainable``/``quantized``,
    the Artifact enum value) of a GlobalWeights row."""
    return (SERVE_DIR / "models" / model_key / str(version_id) / str(weights_id)
            / f"{artifact}.tflite.{_ext}")


def firmware_path(version: str) -> Path:
    return SERVE_DIR / "firmware" / version / f"firmware.bin.{_ext}"


def write_compressed(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compress(data))
