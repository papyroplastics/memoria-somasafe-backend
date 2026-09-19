"""Transport compression for the blobs the gateway serves.

Blobs are stored compressed (common.db) and served as stored; the client
decompresses. Signatures cover the *raw* bytes, so compression happens after
signing and the client verifies after decompressing — it is a pure transport
wrapper, invisible to the signing scheme.
"""

import threading

import zstandard

_LEVEL = 19

_compressor = zstandard.ZstdCompressor(level=_LEVEL)
_decompressor = zstandard.ZstdDecompressor()
_compress_lock = threading.Lock()
_decompress_lock = threading.Lock()


def compress(data: bytes) -> bytes:
    if _compress_lock.acquire(blocking=False):
        try:
            return _compressor.compress(data)
        finally:
            _compress_lock.release()
    return zstandard.ZstdCompressor(level=_LEVEL).compress(data)


def decompress(data: bytes) -> bytes:
    if _decompress_lock.acquire(blocking=False):
        try:
            return _decompressor.decompress(data)
        finally:
            _decompress_lock.release()
    return zstandard.ZstdDecompressor().decompress(data)
