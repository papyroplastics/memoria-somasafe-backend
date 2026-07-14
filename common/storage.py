"""Object store for the blobs the gateway serves (model artifacts, firmware
images, per-user quantization results). Objects live in a MinIO/S3 bucket at
keys derived purely from the DB row that owns them — so a handler can locate
an object from the row it already loaded, with no filename column. The seed
script and the worker are the only writers; the API only ever reads.

Bytes are stored and served compressed: the client decompresses. Signatures in
the DB cover the *raw* bytes, so compression happens after signing and the
client verifies after decompressing.
"""

import io
import uuid

import zstandard
from minio import Minio
from minio.error import S3Error

from common.config import S3_ACCESS_KEY, S3_BUCKET, S3_ENDPOINT_URL, S3_SECRET_KEY, S3_SECURE

_client = Minio(S3_ENDPOINT_URL, access_key=S3_ACCESS_KEY,
                secret_key=S3_SECRET_KEY, secure=S3_SECURE)

_compressor = zstandard.ZstdCompressor(level=19)
_decompressor = zstandard.ZstdDecompressor()
_ext = "zst"

def compress(data: bytes) -> bytes:
    return _compressor.compress(data)


def decompress(data: bytes) -> bytes:
    return _decompressor.decompress(data)


def ensure_bucket() -> None:
    if not _client.bucket_exists(S3_BUCKET):
        _client.make_bucket(S3_BUCKET)


def weights_artifact_key(model_key: str, version_id: int, weights_id: int,
                         artifact: str) -> str:
    """Key for one serving artifact (``artifact`` is ``trainable``/``quantized``,
    the Artifact enum value) of a GlobalWeights row."""
    return f"models/{model_key}/{version_id}/{weights_id}/{artifact}.tflite.{_ext}"


def firmware_key(version: str) -> str:
    return f"firmware/{version}/firmware.bin.{_ext}"


def quantize_result_key(job_id: uuid.UUID) -> str:
    """Key for a QuantizationJob's result. Keyed by the job's own uuid4 id,
    already unguessable, so no separate random token is needed."""
    return f"quantize-results/{job_id}.tflite.{_ext}"


def put_compressed(key: str, data: bytes) -> int:
    """Compress and upload ``data``, returning the compressed size."""
    payload = compress(data)
    _client.put_object(S3_BUCKET, key, io.BytesIO(payload), length=len(payload))
    return len(payload)


def fetch_raw(key: str) -> bytes:
    """The object's bytes as stored — still zstd-compressed, the client
    decompresses (see module docstring)."""
    resp = _client.get_object(S3_BUCKET, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def object_exists(key: str) -> bool:
    try:
        _client.stat_object(S3_BUCKET, key)
        return True
    except S3Error:
        return False


def delete_object(key: str) -> None:
    _client.remove_object(S3_BUCKET, key)
