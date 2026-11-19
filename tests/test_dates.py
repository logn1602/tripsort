import pytest

from tripsort.ingest.dates import OccurrenceFacts, parse_filename_date, resolve_date


@pytest.mark.parametrize(
    ("name", "iso", "tz", "has_time"),
    [
        ("IMG-20260714-WA0001.jpg", "2026-07-14", None, False),
        ("VID-20260714-WA0012.mp4", "2026-07-14", None, False),
        ("WhatsApp Image 2026-07-14 at 10.23.45.jpeg", "2026-07-14T10:23:45", None, True),
        ("WhatsApp Image 2026-07-14 at 6.23.45 PM.jpeg", "2026-07-14T18:23:45", None, True),
        ("WhatsApp Video 2026-07-14 at 12.05.00 AM (1).mp4", "2026-07-14T00:05:00", None, True),
        ("WhatsApp Image 2026-07-14 at 12.05.00 pm.jpeg", "2026-07-14T12:05:00", None, True),
        ("PXL_20260714_082345123.jpg", "2026-07-14T08:23:45.123", "+00:00", True),
        ("PXL_20260714_082345123.MP.jpg", "2026-07-14T08:23:45.123", "+00:00", True),
        ("20260714_102345.jpg", "2026-07-14T10:23:45", None, True),
        ("20260714_102345(0).jpg", "2026-07-14T10:23:45", None, True),
        ("IMG_20260714_102345.jpg", "2026-07-14T10:23:45", None, True),
        ("IMG_20260714_102345_HDR.jpg", "2026-07-14T10:23:45", None, True),
        ("VID_20260714_102345.mp4", "2026-07-14T10:23:45", None, True),
    ],
)
def test_filename_patterns(name: str, iso: str, tz: str | None, has_time: bool) -> None:
    found = parse_filename_date(name)
    assert found is not None
    assert (found.iso, found.tz_offset, found.has_time) == (iso, tz, has_time)


@pytest.mark.parametrize(
    "name",
    [
        "IMG_0412.JPG",
        "photo.jpg",
        "IMG-20261314-WA0001.jpg",  # month 14
        "IMG_20260230_102345.jpg",  # 30 February
        "20260714_256000.jpg",  # hour 25
        "IMG-21500101-WA0001.jpg",  # year out of range
        "WhatsApp Image 2026-07-14 at 13.05.00 PM.jpeg",  # 13 PM
        "Screenshot_20260714-102345.png",
    ],
)
def test_names_without_a_date(name: str) -> None:
    assert parse_filename_date(name) is None


def occ(label: str, path: str, mtime: str | None = None) -> OccurrenceFacts:
    return OccurrenceFacts(label, path, mtime)


def test_exif_wins() -> None:
    resolved = resolve_date(
        "2026-07-14T10:23:45.120", "+02:00", [occ("mine", "IMG_20250101_000000.jpg")]
    )
    assert (resolved.taken_at, resolved.source, resolved.tz_offset) == (
        "2026-07-14T10:23:45.120",
        "exif",
        "+02:00",
    )


def test_filename_with_a_time_beats_a_date_only_name() -> None:
    occurrences = [
        occ("alex", "IMG-20260714-WA0001.jpg"),
        occ("sam", "IMG_20260714_102345.jpg"),
    ]
    resolved = resolve_date(None, None, occurrences)
    assert (resolved.taken_at, resolved.source) == ("2026-07-14T10:23:45", "filename")
    assert resolve_date(None, None, occurrences[::-1]) == resolved  # order of zips is irrelevant


def test_zip_time_is_the_last_resort_and_the_earliest_wins() -> None:
    resolved = resolve_date(
        None,
        None,
        [occ("b", "a.jpg", "2026-07-15T09:00:00"), occ("a", "b.jpg", "2026-07-14T09:00:00")],
    )
    assert (resolved.taken_at, resolved.source, resolved.tz_offset) == (
        "2026-07-14T09:00:00",
        "zip_mtime",
        None,
    )


def test_no_date_at_all() -> None:
    resolved = resolve_date(None, None, [occ("a", "photo.jpg")])
    assert (resolved.taken_at, resolved.source) == (None, None)
