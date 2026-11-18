"""SHA-256 identity and 64-bit perceptual hashes."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import imagehash
from PIL import Image

PHASH_ALGORITHM = "imagehash.phash, hash_size=8 (64 bits), on the EXIF-oriented image"


@dataclass(frozen=True)
class HashedBytes:
    sha256: str
    size: int
    data: bytes | None  # the bytes themselves, when the caller asked to keep them


def hash_chunks(chunks: Iterable[bytes], *, keep: bool) -> HashedBytes:
    """Hash a stream of chunks; keep the bytes only when they will be decoded next."""
    digest = hashlib.sha256()
    size = 0
    buffer = bytearray() if keep else None
    for chunk in chunks:
        digest.update(chunk)
        size += len(chunk)
        if buffer is not None:
            buffer += chunk
    return HashedBytes(digest.hexdigest(), size, bytes(buffer) if buffer is not None else None)


def phash_hex(image: Image.Image) -> str:
    """The 64-bit DCT perceptual hash as 16 hex digits."""
    return str(imagehash.phash(image, hash_size=8))


def phash_to_int(value: str) -> int:
    return int(value, 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
