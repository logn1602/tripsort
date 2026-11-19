"""Export file names: ``YYYYMMDD_HHMMSS_<source>_<original-name>``, safe on every OS.

Names are compared case-insensitively and after Unicode normalization, because Windows
and macOS treat ``IMG_1.JPG`` and ``img_1.jpg`` (or NFC and NFD forms) as the same file.
"""

import re
import unicodedata
from pathlib import PurePosixPath

MAX_NAME = 100  # characters per name: duplicates/<name>/<name> stays short on Windows
MAX_LABEL = 32
MAX_FOLDER = 64  # a person's folder name
_MAX_EXTENSION = 10
_FORBIDDEN = frozenset('<>:"/\\|?*')
_WHITESPACE = re.compile(r"\s+")
_RESERVED = frozenset(  # names Windows reserves for devices, with or without an extension
    {"con", "prn", "aux", "nul"}
    | {f"{port}{n}" for port in ("com", "lpt") for n in "0123456789¹²³"}
)


def sanitize(text: str) -> str:
    """Replace characters Windows forbids and drop trailing dots and spaces."""
    text = unicodedata.normalize("NFC", text)
    cleaned = "".join(
        "_" if ch in _FORBIDDEN or ord(ch) < 32 or ord(ch) == 127 else ch for ch in text
    )
    return cleaned.strip().rstrip(". ")


def stamp(taken_at: str | None) -> str:
    """'2026-07-14T10:23:45.123' -> '20260714_102345'; a date alone gets '000000'."""
    if not taken_at or len(taken_at) < 10:
        return "00000000_000000"
    day = taken_at[:10].replace("-", "")
    clock = taken_at[11:19].replace(":", "") if len(taken_at) >= 19 else "000000"
    return f"{day}_{clock}"


def label_part(label: str) -> str:
    cleaned = _WHITESPACE.sub("-", sanitize(label))[:MAX_LABEL].rstrip(". ")
    return cleaned or "zip"


def split_extension(name: str) -> tuple[str, str]:
    suffix = PurePosixPath(name).suffix
    if not suffix or len(suffix) > _MAX_EXTENSION or suffix == name:
        return name, ""
    return name[: -len(suffix)], suffix


def export_name(taken_at: str | None, label: str, member_path: str) -> str:
    original = sanitize(PurePosixPath(member_path).name) or "file"
    stem, extension = split_extension(original)
    prefix = f"{stamp(taken_at)}_{label_part(label)}_"
    room = max(8, MAX_NAME - len(prefix) - len(extension))
    stem = stem[:room].rstrip(". ") or "file"
    return f"{prefix}{stem}{extension}"


def person_folder(name: str) -> str:
    """The folder for a person's name: 'Sam / Alex' -> 'Sam _ Alex', 'Con' -> 'Con_'."""
    cleaned = _WHITESPACE.sub(" ", sanitize(name))[:MAX_FOLDER].rstrip(". ") or "person"
    if cleaned.split(".", 1)[0].strip().casefold() in _RESERVED:
        cleaned += "_"
    return cleaned


def numbered(name: str, n: int) -> str:
    """The n-th alternative for a name that is taken: 'a.jpg' -> 'a_1.jpg'."""
    stem, extension = split_extension(name)
    return f"{stem}_{n}{extension}"


def fold_key(relpath: str) -> str:
    """The form two paths share when a case-insensitive file system sees them as one."""
    return unicodedata.normalize("NFC", relpath).casefold()
