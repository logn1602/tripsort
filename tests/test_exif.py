import io

import pytest
from factories import encode, exif_block, jpeg, photo
from PIL import Image

from tripsort.ingest.exif import (
    NO_EXIF,
    parse_exif_datetime,
    parse_exposure,
    parse_iso,
    parse_offset,
    read_exif,
)


def exif_of(data: bytes):  # type: ignore[no-untyped-def]
    with Image.open(io.BytesIO(data)) as image:
        return read_exif(image)


def test_full_camera_exif() -> None:
    data = jpeg(
        photo(1, (64, 48)),
        exif=exif_block(
            datetime="2026:07:14 10:23:45",
            subsec="123",
            offset="+05:30",
            make="Apple",
            model="iPhone 15",
            lens="iPhone 15 front camera 2.69mm f/1.9",
            orientation=6,
            gps=(48.8584, 2.2945),
            exposure=(1, 120),
            iso=200,
        ),
    )
    info = exif_of(data)
    assert info.exposure_time == pytest.approx(1 / 120)
    assert info.iso == 200
    assert info.datetime == "2026-07-14T10:23:45.123"
    assert info.offset == "+05:30"
    assert (info.make, info.model) == ("Apple", "iPhone 15")
    assert info.lens == "iPhone 15 front camera 2.69mm f/1.9"
    assert info.orientation == 6
    assert info.swaps_axes
    assert info.has_camera_exif
    assert info.gps_lat == pytest.approx(48.8584, abs=1e-4)
    assert info.gps_lon == pytest.approx(2.2945, abs=1e-4)


def test_southern_and_western_coordinates_are_negative() -> None:
    info = exif_of(jpeg(photo(1, (64, 48)), exif=exif_block(gps=(-33.8568, -151.2153))))
    assert info.gps_lat == pytest.approx(-33.8568, abs=1e-4)
    assert info.gps_lon == pytest.approx(-151.2153, abs=1e-4)


def test_zero_zero_gps_means_no_fix() -> None:
    info = exif_of(jpeg(photo(1, (64, 48)), exif=exif_block(gps=(0.0, 0.0))))
    assert (info.gps_lat, info.gps_lon) == (None, None)


def test_digitized_time_is_the_fallback() -> None:
    info = exif_of(jpeg(photo(1, (64, 48)), exif=exif_block(digitized="2026:07:14 08:00:00")))
    assert info.datetime == "2026-07-14T08:00:00"


def test_modify_date_is_read_to_the_second() -> None:
    taken = "2026:07:14 10:23:45"
    camera = exif_block(datetime=taken, subsec="120", modified=taken)
    info = exif_of(jpeg(photo(1, (64, 48)), exif=camera))
    assert (info.datetime, info.modified) == ("2026-07-14T10:23:45.120", "2026-07-14T10:23:45")
    no_modify_date = exif_block(datetime="2026:07:14 10:23:45")
    assert exif_of(jpeg(photo(1, (64, 48)), exif=no_modify_date)).modified is None
    editor_only = exif_of(jpeg(photo(1, (64, 48)), exif=exif_block(modified="2026:11:02 18:05:12")))
    assert editor_only.modified == "2026-11-02T18:05:12"
    assert not editor_only.has_camera_exif  # a modify date alone is not camera EXIF


def test_orientation_alone_is_not_camera_exif() -> None:
    info = exif_of(jpeg(photo(1, (64, 48)), exif=exif_block(orientation=1)))
    assert not info.has_camera_exif


def test_no_exif() -> None:
    assert exif_of(jpeg(photo(1, (64, 48)))) == NO_EXIF
    assert exif_of(encode(photo(1, (64, 48)), "PNG")) == NO_EXIF


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")  # raised by this test's Image.open
def test_broken_exif_counts_as_none() -> None:
    data = jpeg(photo(1, (64, 48)), exif=b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\xff\xff")
    assert exif_of(data) == NO_EXIF


@pytest.mark.parametrize(
    ("value", "subsec", "expected"),
    [
        ("2026:07:14 10:23:45", None, "2026-07-14T10:23:45"),
        ("2026:07:14 10:23:45", "5", "2026-07-14T10:23:45.5"),
        ("2026:07:14 10:23:45", "1234567", "2026-07-14T10:23:45.123456"),
        ("2026-07-14 10:23:45", None, "2026-07-14T10:23:45"),
        ("2026:07:14 10:23:45\x00", " 42 ", "2026-07-14T10:23:45.42"),
        (b"2026:07:14 10:23:45", None, "2026-07-14T10:23:45"),
        ("0000:00:00 00:00:00", None, None),
        ("    :  :     :  :  ", None, None),
        ("1800:01:01 00:00:00", None, None),
        ("2026:02:30 10:00:00", None, None),
        ("garbage", None, None),
        (None, None, None),
        (12345, None, None),
    ],
)
def test_parse_exif_datetime(value: object, subsec: object, expected: str | None) -> None:
    assert parse_exif_datetime(value, subsec) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+05:30", "+05:30"),
        ("-08:00", "-08:00"),
        (" +00:00 ", "+00:00"),
        ("+15:00", None),
        ("+05:75", None),
        ("05:30", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_offset(value: object, expected: str | None) -> None:
    assert parse_offset(value) == expected


def test_iso_stored_as_several_values() -> None:
    info = exif_of(jpeg(photo(1, (64, 48)), exif=exif_block(iso=(400, 400))))
    assert info.iso == 400


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.008, 0.008), (30, 30.0), (0, None), (-1, None), (7200, None), ("x", None), (None, None)],
)
def test_parse_exposure(value: object, expected: float | None) -> None:
    assert parse_exposure(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100, 100),
        (3200.0, 3200),
        ((50, 50), 50),
        ((), None),
        (0, None),
        (True, None),
        ("100", None),
    ],
)
def test_parse_iso(value: object, expected: int | None) -> None:
    assert parse_iso(value) == expected
