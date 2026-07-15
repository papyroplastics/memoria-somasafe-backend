"""Transport compression for the blobs the gateway serves.

Blobs are stored compressed (common.db) and served as stored; the client
decompresses. Signatures cover the *raw* bytes, so compression happens after
signing and the client verifies after decompressing — it is a pure transport
wrapper, invisible to the signing scheme.
"""

import zstandard

_compressor = zstandard.ZstdCompressor(level=19)
_decompressor = zstandard.ZstdDecompressor()


def compress(data: bytes) -> bytes:
    return _compressor.compress(data)


def decompress(data: bytes) -> bytes:
    return _decompressor.decompress(data)
