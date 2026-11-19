import zipfile
from pathlib import Path

import pytest
from factories import make_zip, patch_entry

from tripsort.errors import ZipError
from tripsort.ingest.zip_reader import (
    SKIP_CORRUPT,
    SKIP_EMPTY,
    SKIP_ENCRYPTED,
    SKIP_JUNK,
    SKIP_UNSUPPORTED_COMPRESSION,
    SKIP_UNSUPPORTED_TYPE,
    ZipSource,
    entry_mtime,
    is_junk,
    media_kind,
    skip_hint,
)


@pytest.mark.parametrize(
    "path",
    [
        "__MACOSX/DCIM/._IMG_0001.JPG",
        "DCIM/.DS_Store",
        "._IMG_0001.JPG",
        "DCIM/.thumbnails/123.jpg",
        "Pictures/.trashed-1700000000-IMG_1.jpg",
        "Thumbs.db",
        "folder/desktop.ini",
        "/",
    ],
)
def test_junk(path: str) -> None:
    assert is_junk(path)


@pytest.mark.parametrize("path", ["DCIM/IMG_0001.JPG", "trip/day 1/photo.heic", "a.b/c.png"])
def test_not_junk(path: str) -> None:
    assert not is_junk(path)


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("a.jpg", "image"),
        ("A.JPEG", "image"),
        ("b.jfif", "image"),
        ("c.HEIC", "image"),
        ("d.heif", "image"),
        ("e.png", "image"),
        ("f.webp", "image"),
        ("g.MP4", "video"),
        ("h.mov", "video"),
        ("i.m4v", "video"),
        ("j.dng", None),
        ("k.gif", None),
        ("noext", None),
    ],
)
def test_media_kind(path: str, kind: str | None) -> None:
    assert media_kind(path) == kind


def test_entry_mtime() -> None:
    assert entry_mtime((1980, 1, 1, 0, 0, 0)) is None  # the DOS epoch means "unset"
    assert entry_mtime((2026, 7, 14, 10, 23, 44)) == "2026-07-14T10:23:44"
    assert entry_mtime((2026, 13, 1, 0, 0, 0)) is None


def test_members_are_classified(tmp_path: Path) -> None:
    path = make_zip(
        tmp_path / "t.zip",
        [
            ("DCIM/", b""),
            ("DCIM/IMG_0001.JPG", b"jpeg bytes"),
            ("DCIM/nested/deeper/IMG_0002.png", b"png bytes"),
            ("__MACOSX/DCIM/._IMG_0001.JPG", b"fork"),
            ("empty.jpg", b""),
            ("notes.txt", b"text"),
            ("clip.mov", b"video"),
        ],
    )
    with ZipSource(path) as source:
        by_path = {m.path: m for m in source.members}
        assert "DCIM/" not in by_path  # directories are not members
        assert by_path["DCIM/IMG_0001.JPG"].skip is None
        assert by_path["DCIM/IMG_0001.JPG"].kind == "image"
        assert by_path["DCIM/IMG_0001.JPG"].index == 1  # position in the central directory
        assert by_path["DCIM/nested/deeper/IMG_0002.png"].name == "IMG_0002.png"
        assert by_path["__MACOSX/DCIM/._IMG_0001.JPG"].skip == SKIP_JUNK
        assert by_path["empty.jpg"].skip == SKIP_EMPTY
        assert by_path["notes.txt"].skip == SKIP_UNSUPPORTED_TYPE
        assert by_path["notes.txt"].detail == ".txt"
        assert by_path["clip.mov"].kind == "video"
        assert source.bytes == path.stat().st_size


def test_duplicate_names_in_one_zip_are_two_members(tmp_path: Path) -> None:
    path = tmp_path / "dup.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("IMG_0001.jpg", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("IMG_0001.jpg", b"second")
    with ZipSource(path) as source:
        assert [m.path for m in source.members] == ["IMG_0001.jpg", "IMG_0001.jpg"]
        assert [m.index for m in source.members] == [0, 1]
        assert [b"".join(source.iter_chunks(m)) for m in source.members] == [b"first", b"second"]


def test_names_without_the_utf8_flag(tmp_path: Path) -> None:
    raw_utf8 = "café.jpg".encode()  # UTF-8 written by a tool that did not set the flag
    raw_cp437 = "é.jpg".encode("cp437")  # an old Windows zip
    data = make_zip(
        tmp_path / "names.zip",
        [("x" * len(raw_utf8), b"one"), ("y" * len(raw_cp437), b"two"), ("é_flagged.jpg", b"3")],
    ).read_bytes()
    data = patch_entry(data, "x" * len(raw_utf8), new_name=raw_utf8)
    data = patch_entry(data, "y" * len(raw_cp437), new_name=raw_cp437)
    path = tmp_path / "patched.zip"
    path.write_bytes(data)
    with ZipSource(path) as source:
        utf8, cp437, flagged = source.members
        assert (utf8.path, utf8.raw_name) == ("café.jpg", raw_utf8)
        assert (cp437.path, cp437.raw_name) == ("é.jpg", raw_cp437)
        assert (flagged.path, flagged.raw_name) == ("é_flagged.jpg", None)


def test_backslash_separators_are_normalized(tmp_path: Path) -> None:
    data = make_zip(tmp_path / "a.zip", [("DCIM/IMG_1.jpg", b"x")]).read_bytes()
    path = tmp_path / "b.zip"
    path.write_bytes(patch_entry(data, "DCIM/IMG_1.jpg", new_name=b"DCIM\\IMG_1.jpg"))
    with ZipSource(path) as source:
        assert source.members[0].path == "DCIM/IMG_1.jpg"


def test_entries_that_cannot_be_read(tmp_path: Path) -> None:
    data = make_zip(
        tmp_path / "a.zip",
        [("locked.jpg", b"secret bytes"), ("deflate64.jpg", b"some bytes"), ("ok.jpg", b"fine")],
    ).read_bytes()
    data = patch_entry(data, "locked.jpg", set_flags=0x1)
    data = patch_entry(data, "deflate64.jpg", method=9)
    path = tmp_path / "b.zip"
    path.write_bytes(data)
    with ZipSource(path) as source:
        locked, deflate64, ok = source.members
        assert locked.skip == SKIP_ENCRYPTED
        assert deflate64.skip == SKIP_UNSUPPORTED_COMPRESSION
        assert deflate64.detail == "Deflate64 (method 9)"
        assert ok.skip is None


def test_skip_hints_say_what_to_do() -> None:
    assert "without encryption" in (skip_hint(SKIP_ENCRYPTED, None) or "")
    hint = skip_hint(SKIP_UNSUPPORTED_COMPRESSION, "Deflate64 (method 9)") or ""
    assert "Deflate64" in hint
    assert "standard Deflate" in hint
    assert "again" in (skip_hint(SKIP_CORRUPT, "Bad CRC-32") or "")
    assert skip_hint(SKIP_JUNK, None) is None


def test_fingerprint_identifies_the_zip(tmp_path: Path) -> None:
    a = make_zip(tmp_path / "a.zip", [("x.jpg", b"one"), ("y.jpg", b"two")])
    copy = tmp_path / "renamed.zip"
    copy.write_bytes(a.read_bytes())
    b = make_zip(tmp_path / "b.zip", [("x.jpg", b"one"), ("y.jpg", b"TWO")])
    with ZipSource(a) as sa, ZipSource(copy) as sc, ZipSource(b) as sb:
        assert sa.fingerprint == sc.fingerprint
        assert sa.fingerprint != sb.fingerprint
        assert len(sa.fingerprint) == 64


def test_missing_or_invalid_zip(tmp_path: Path) -> None:
    with pytest.raises(ZipError, match="no such file"):
        ZipSource(tmp_path / "missing.zip")
    not_zip = tmp_path / "photo.zip"
    not_zip.write_bytes(b"this is not a zip")
    with pytest.raises(ZipError, match="not a readable zip"):
        ZipSource(not_zip)
