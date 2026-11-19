import io
import warnings

import numpy as np
import pytest
from factories import camera_exif, encode, exif_block, heic, jpeg, photo, whatsapp_copy
from PIL import Image

from tripsort.ingest.decode import (
    UnreadableImageError,
    analyze_image,
    draft_box,
    open_image,
)
from tripsort.ingest.hashing import hamming, phash_hex, phash_to_int


def distance(a: str | None, b: str | None) -> int:
    assert a is not None
    assert b is not None
    return hamming(phash_to_int(a), phash_to_int(b))


def test_jpeg() -> None:
    result = analyze_image(jpeg(photo(1), exif=camera_exif()))
    assert (result.format, result.width, result.height) == ("JPEG", 1200, 900)
    assert result.decoded
    assert result.phash is not None
    assert len(result.phash) == 16
    assert result.exif.model == "Pixel 7"
    assert result.note is None


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_png_and_webp(fmt: str) -> None:
    image = photo(1)
    result = analyze_image(encode(image, fmt))
    assert (result.format, result.width, result.height) == (fmt, 1200, 900)
    assert distance(result.phash, phash_hex(image)) <= 2


def test_heic_decodes_through_pi_heif() -> None:
    result = analyze_image(heic(photo(3, (480, 360)), exif=camera_exif(model="iPhone 15")))
    assert (result.format, result.width, result.height) == ("HEIF", 480, 360)
    assert result.decoded
    assert result.phash is not None
    assert result.exif.model == "iPhone 15"
    assert result.exif.datetime == "2026-07-14T10:23:45"


def test_the_registered_heif_decoder_is_pi_heif() -> None:
    open_image(jpeg(photo(1, (32, 24))))  # registers the plugins
    assert Image.OPEN["HEIF"][0].__module__.startswith("pi_heif")


def test_rotated_heic_reports_its_displayed_size() -> None:
    result = analyze_image(heic(photo(3, (480, 360)), exif=exif_block(orientation=6)))
    assert (result.width, result.height) == (360, 480)
    assert result.exif.orientation == 1  # the decoder applied the rotation itself


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied_before_hashing(orientation: int) -> None:
    stored = photo(7, (1200, 800))
    data = jpeg(stored, exif=exif_block(orientation=orientation))
    result = analyze_image(data)
    rotated = orientation in {5, 6, 7, 8}
    assert (result.width, result.height) == ((800, 1200) if rotated else (1200, 800))
    # A messenger copy has the rotation baked in and no EXIF; it must hash the same.
    copy = analyze_image(whatsapp_copy(data))
    assert copy.exif.orientation == 1
    assert distance(result.phash, copy.phash) <= 2


def test_draft_box_keeps_the_aspect_ratio() -> None:
    assert draft_box((4000, 3000), 1600) == (1600, 1200)
    assert draft_box((3000, 4000), 1600) == (1200, 1600)
    assert draft_box((1600, 900), 1600) is None
    assert draft_box((12000, 10), 1600) == (1600, 2)


def test_large_jpeg_uses_draft_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    data = jpeg(photo(4, (4000, 3000)))
    # 12 MP is over a 4 MP limit, but draft mode decodes it at 2000x1500 = 3 MP.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4_000_000)
    result = analyze_image(data)
    assert result.decoded
    assert (result.width, result.height) == (4000, 3000)
    assert distance(result.phash, phash_hex(photo(4, (4000, 3000)))) <= 2


def test_jpeg_too_large_even_for_draft_mode_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    result = analyze_image(jpeg(photo(4, (4000, 3000))))  # 3 MP after draft > 1 MP
    assert not result.decoded
    assert result.phash is None


def test_oversized_non_jpeg_is_passed_through_not_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = encode(photo(4, (2000, 1500)), "PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    result = analyze_image(data)
    assert not result.decoded
    assert result.phash is None
    assert (result.format, result.width, result.height) == ("PNG", 2000, 1500)
    assert result.note is not None
    assert "1 MP decode limit" in result.note


def test_no_pixel_limit_decodes_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    assert analyze_image(encode(photo(4, (2000, 1500)), "PNG")).decoded


def test_other_formats_pillow_knows_still_work() -> None:
    result = analyze_image(encode(photo(5, (300, 200)), "GIF"))  # e.g. a GIF named .jpg
    assert result.format == "GIF"
    assert result.decoded


def test_other_formats_keep_pillows_bomb_check(monkeypatch: pytest.MonkeyPatch) -> None:
    data = encode(photo(5, (300, 200)), "GIF")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20_000)  # 60,000 px > 2 x limit
    result = analyze_image(data)
    assert not result.decoded
    assert result.note is not None
    assert result.note.startswith("not decoded")


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [("CMYK", "JPEG"), ("L", "JPEG"), ("P", "PNG"), ("RGBA", "PNG"), ("I;16", "PNG"), ("1", "PNG")],
)
def test_unusual_modes(mode: str, fmt: str) -> None:
    image = photo(6, (400, 300))
    if mode == "I;16":
        gray = np.asarray(image.convert("L"), dtype=np.uint16) * 257
        image = Image.fromarray(gray)
        assert image.mode in {"I;16", "I"}
    else:
        image = image.convert(mode)
    result = analyze_image(encode(image, fmt))
    assert result.decoded
    assert result.phash is not None


def test_truncated_jpeg_is_unreadable() -> None:
    data = jpeg(photo(1))
    with pytest.raises(UnreadableImageError, match="truncated"):
        analyze_image(data[: len(data) // 3])


@pytest.mark.parametrize(
    "data",
    [b"not an image at all", b"\xff\xd8\xff\xe0garbage", b"\x89PNG\r\n\x1a\ngarbage", b""],
)
def test_garbage_is_unreadable(data: bytes) -> None:
    with pytest.raises(UnreadableImageError):
        analyze_image(data)


def test_decoding_emits_no_warnings() -> None:
    broken_exif = jpeg(photo(1, (64, 48)), exif=b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\xff\xff")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyze_image(broken_exif)
    assert result.decoded
    assert caught == []


def test_a_failing_phash_keeps_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    import tripsort.ingest.decode as decode

    def broken(image: Image.Image) -> str:
        raise ValueError("no luck")

    monkeypatch.setattr(decode, "phash_hex", broken)
    result = analyze_image(jpeg(photo(1)))
    assert result.decoded
    assert result.phash is None
    assert result.note == "no pHash: ValueError: no luck"


def test_open_image_does_not_decode_pixels() -> None:
    image = open_image(jpeg(photo(1)))
    assert image.size == (1200, 900)
    assert isinstance(image.fp, io.BytesIO)  # still lazy: nothing loaded yet
