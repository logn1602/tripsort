"""Read zips in place: list entries, filter junk, decode names, and fingerprint the zip.

Nothing is ever extracted to disk; members are streamed with ``zipfile``.
"""

import contextlib
import hashlib
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import IO, Literal, Self

from tripsort.errors import ZipError

MediaKind = Literal["image", "video"]

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".heic", ".heif", ".hif"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v"})
JUNK_NAMES = frozenset({"thumbs.db", "ehthumbs.db", "desktop.ini"})

# Reasons an entry is skipped. The first three are deliberate; the rest mean the entry's
# bytes could not be read, so the run is only a partial success.
SKIP_JUNK = "junk"
SKIP_EMPTY = "empty"
SKIP_UNSUPPORTED_TYPE = "unsupported_type"
SKIP_ENCRYPTED = "encrypted"
SKIP_UNSUPPORTED_COMPRESSION = "unsupported_compression"
SKIP_CORRUPT = "corrupt"
READ_FAILURES = frozenset({SKIP_ENCRYPTED, SKIP_UNSUPPORTED_COMPRESSION, SKIP_CORRUPT})

_UTF8_FLAG = 0x800
_ENCRYPTED_FLAG = 0x1
_DOS_EPOCH = (1980, 1, 1, 0, 0, 0)
_SUPPORTED_COMPRESSION = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}
)
_COMPRESSION_NAMES = {
    6: "Implode",
    9: "Deflate64",
    10: "PKWARE DCL Implode",
    93: "Zstandard",
    95: "XZ",
    96: "JPEG variant",
    97: "WavPack",
    98: "PPMd",
    99: "AES encryption",
}


@dataclass(frozen=True)
class Member:
    """One file entry of a zip (directories are left out)."""

    index: int  # position in the zip's central directory
    info: zipfile.ZipInfo
    path: str  # decoded name with '/' separators
    raw_name: bytes | None  # original bytes, kept when the zip did not mark the name UTF-8
    size: int
    mtime: str | None  # the entry's timestamp as ISO 8601 (no zone), None if unset
    kind: MediaKind | None
    skip: str | None  # why the entry is not read, or None
    detail: str | None = None

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name


def skip_hint(reason: str, detail: str | None) -> str | None:
    """What the user can do about an entry tripsort could not read, or None."""
    if reason == SKIP_ENCRYPTED:
        return (
            "The entry is password-protected. Extract the zip with its password, then zip "
            "the files again without encryption."
        )
    if reason == SKIP_UNSUPPORTED_COMPRESSION:
        method = detail or "an unsupported method"
        return (
            f"The entry is compressed with {method}, which Python's zipfile cannot read. "
            "Extract the zip, then zip the files again with standard Deflate compression "
            "(the default in Windows, macOS, and most zip tools)."
        )
    if reason == SKIP_CORRUPT:
        return "The entry is damaged. Download or create the zip again."
    return None


def decode_name(info: zipfile.ZipInfo) -> tuple[str, bytes | None]:
    """Return the entry name with '/' separators, plus the raw bytes when they matter.

    Names flagged UTF-8 are used as they are. Otherwise ``zipfile`` decodes them as cp437;
    many tools write UTF-8 without setting the flag, so valid UTF-8 is preferred, and the
    raw bytes are kept for the manifest.
    """
    name = info.orig_filename
    raw: bytes | None = None
    if not info.flag_bits & _UTF8_FLAG:
        try:
            encoded = name.encode("cp437")
        except UnicodeEncodeError:  # pragma: no cover - only with a custom metadata_encoding
            encoded = None
        if encoded is not None and not encoded.isascii():
            raw = encoded
            with contextlib.suppress(UnicodeDecodeError):
                name = encoded.decode("utf-8")
    name = name.split("\x00", 1)[0].replace("\\", "/")
    return name, raw


def is_junk(path: str) -> bool:
    """macOS resource forks, hidden files and folders, and Windows thumbnail caches."""
    parts = [part for part in path.split("/") if part]
    if not parts:
        return True
    if "__MACOSX" in parts[:-1] or any(part.startswith(".") for part in parts):
        return True
    return parts[-1].lower() in JUNK_NAMES


def media_kind(path: str) -> MediaKind | None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def entry_mtime(date_time: tuple[int, int, int, int, int, int]) -> str | None:
    if tuple(date_time) == _DOS_EPOCH:
        return None
    try:
        return datetime(*date_time).isoformat()
    except ValueError:
        return None


def classify(index: int, info: zipfile.ZipInfo) -> Member:
    path, raw = decode_name(info)
    kind = media_kind(path)
    skip: str | None = None
    detail: str | None = None
    if is_junk(path):
        skip = SKIP_JUNK
    elif info.file_size == 0:
        skip = SKIP_EMPTY
    elif kind is None:
        skip = SKIP_UNSUPPORTED_TYPE
        detail = PurePosixPath(path).suffix.lower() or "no extension"
    elif info.flag_bits & _ENCRYPTED_FLAG:
        skip = SKIP_ENCRYPTED
    elif info.compress_type not in _SUPPORTED_COMPRESSION:
        skip = SKIP_UNSUPPORTED_COMPRESSION
        method = _COMPRESSION_NAMES.get(info.compress_type, "an unknown method")
        detail = f"{method} (method {info.compress_type})"
    return Member(
        index=index,
        info=info,
        path=path,
        raw_name=raw,
        size=info.file_size,
        mtime=entry_mtime(info.date_time),
        kind=kind,
        skip=skip,
        detail=detail,
    )


def fingerprint(infos: list[zipfile.ZipInfo]) -> str:
    """SHA-256 of the central directory listing: names, CRC-32s, sizes, offsets, times.

    Reading it costs milliseconds, unlike hashing the whole zip, and any change to an
    entry changes it.
    """
    digest = hashlib.sha256()
    for info in infos:
        fields = (
            info.orig_filename,
            info.CRC,
            info.file_size,
            info.compress_size,
            info.header_offset,
            info.date_time,
            info.flag_bits,
            info.compress_type,
        )
        digest.update(repr(fields).encode("utf-8", "surrogateescape"))
        digest.update(b"\n")
    return digest.hexdigest()


class ZipSource:
    """An open input zip. Use as a context manager, or call ``close()``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.is_file():
            raise ZipError(f"{path}: no such file")
        try:
            self._zip = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError) as exc:
            raise ZipError(f"{path}: not a readable zip file ({exc})") from exc
        infos = self._zip.infolist()
        self.fingerprint = fingerprint(infos)
        self.members = [classify(i, info) for i, info in enumerate(infos) if not info.is_dir()]
        self.bytes = path.stat().st_size

    def open(self, member: Member) -> IO[bytes]:
        return self._zip.open(member.info)

    def iter_chunks(self, member: Member, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        with self.open(member) as fp:
            while chunk := fp.read(chunk_size):
                yield chunk

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
