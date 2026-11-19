import hashlib
import io

from factories import jpeg, photo
from PIL import Image

from tripsort.ingest.hashing import hamming, hash_chunks, phash_hex, phash_to_int


def test_streamed_sha256_matches_hashlib() -> None:
    data = bytes(range(256)) * 10_000
    chunks = [data[i : i + 65536] for i in range(0, len(data), 65536)]
    hashed = hash_chunks(chunks, keep=False)
    assert hashed.sha256 == hashlib.sha256(data).hexdigest()
    assert hashed.size == len(data)
    assert hashed.data is None
    assert hash_chunks(chunks, keep=True).data == data


def test_empty_stream() -> None:
    hashed = hash_chunks([], keep=True)
    assert hashed.sha256 == hashlib.sha256(b"").hexdigest()
    assert (hashed.size, hashed.data) == (0, b"")


def test_phash_is_16_hex_digits() -> None:
    value = phash_hex(photo(1))
    assert len(value) == 16
    assert phash_to_int(value) == int(value, 16)


def test_phash_distances() -> None:
    original = photo(1)
    recompressed = Image.open(io.BytesIO(jpeg(original.resize((600, 450)), quality=60)))
    same = phash_to_int(phash_hex(original))
    assert hamming(same, phash_to_int(phash_hex(original.copy()))) == 0
    assert hamming(same, phash_to_int(phash_hex(recompressed))) <= 2
    assert hamming(same, phash_to_int(phash_hex(photo(2)))) > 12


def test_hamming() -> None:
    assert hamming(0, 0) == 0
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(0, 2**64 - 1) == 64
