import unicodedata

import pytest

from tripsort.export.naming import (
    MAX_NAME,
    export_name,
    fold_key,
    label_part,
    numbered,
    sanitize,
    split_extension,
    stamp,
)


@pytest.mark.parametrize(
    ("taken_at", "expected"),
    [
        ("2026-07-14T10:23:45.123", "20260714_102345"),
        ("2026-07-14T10:23:45", "20260714_102345"),
        ("2026-07-14", "20260714_000000"),
        (None, "00000000_000000"),
        ("", "00000000_000000"),
    ],
)
def test_stamp(taken_at: str | None, expected: str) -> None:
    assert stamp(taken_at) == expected


def test_export_name() -> None:
    name = export_name("2026-07-14T10:23:45", "sam", "DCIM/100APPLE/IMG_0412.JPG")
    assert name == "20260714_102345_sam_IMG_0412.JPG"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('a<b>c:d"e/f\\g|h?i*j.jpg', "a_b_c_d_e_f_g_h_i_j.jpg"),
        ("tab\there.jpg", "tab_here.jpg"),
        ("trailing dots... ", "trailing dots"),
        ("  spaced.jpg  ", "spaced.jpg"),
        (unicodedata.normalize("NFD", "café.jpg"), "café.jpg"),
    ],
)
def test_sanitize(text: str, expected: str) -> None:
    assert sanitize(text) == expected


def test_label_part() -> None:
    assert label_part("Sam's phone") == "Sam's-phone"
    assert label_part("   ") == "zip"
    assert label_part("x" * 100) == "x" * 32


def test_long_names_are_shortened_but_keep_their_extension() -> None:
    name = export_name("2026-07-14T10:23:45", "sam", "a" * 300 + ".jpeg")
    assert len(name) <= MAX_NAME
    assert name.endswith("a.jpeg")


def test_empty_or_odd_original_names() -> None:
    assert export_name(None, "sam", "dir/...") == "00000000_000000_sam_file"
    assert export_name(None, "sam", "noextension") == "00000000_000000_sam_noextension"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("IMG_1.jpg", ("IMG_1", ".jpg")),
        ("archive.tar.gz", ("archive.tar", ".gz")),
        ("README", ("README", "")),
        ("photo.averyverylongsuffix", ("photo.averyverylongsuffix", "")),
    ],
)
def test_split_extension(name: str, expected: tuple[str, str]) -> None:
    assert split_extension(name) == expected


def test_numbered() -> None:
    assert numbered("20260714_102345_sam_IMG_1.jpg", 1) == "20260714_102345_sam_IMG_1_1.jpg"
    assert numbered("README", 2) == "README_2"


def test_fold_key_matches_case_and_unicode_variants() -> None:
    assert fold_key("all/IMG_1.JPG") == fold_key("all/img_1.jpg")
    assert fold_key(unicodedata.normalize("NFD", "all/café.jpg")) == fold_key("all/CAFÉ.jpg")
    assert fold_key("all/a.jpg") != fold_key("all/b.jpg")
