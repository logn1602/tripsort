from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import camera_exif, heic, jpeg, make_zip, photo, whatsapp_copy

from tripsort.api import Project


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Project]:
    with Project.open(tmp_path / "sorted") as proj:
        yield proj


@pytest.fixture
def trip(tmp_path: Path) -> dict[str, Path]:
    """Two zips from a small trip: exact duplicates across zips, a WhatsApp copy, a HEIC,
    junk, an unsupported file, a video, and a truncated JPEG."""
    a = jpeg(photo(1), exif=camera_exif("2026:07:14 10:23:45", subsec="120"))
    b = jpeg(photo(2), exif=camera_exif("2026:07:14 11:00:00", orientation=6))
    c = heic(
        photo(3, (480, 360)),
        exif=camera_exif("2026:07:15 09:00:00", make="Apple", model="iPhone 15"),
    )
    mine = make_zip(
        tmp_path / "in" / "mine.zip",
        [
            ("DCIM/PXL_20260714_082345120.jpg", a),
            ("DCIM/PXL_20260714_090000000.jpg", b),
            ("DCIM/IMG_0001.HEIC", c),
            ("__MACOSX/DCIM/._PXL_20260714_082345120.jpg", b"junk"),
            (".DS_Store", b"junk"),
            ("notes.txt", b"hello"),
            ("DCIM/VID_20260714_101010.mp4", b"\x00\x00\x00\x18ftypmp42" + bytes(4000)),
            ("DCIM/broken.jpg", a[:3000]),
        ],
    )
    sam = make_zip(
        tmp_path / "in" / "sam.zip",
        [
            ("WhatsApp Images/IMG-20260714-WA0001.jpg", whatsapp_copy(a)),
            ("Camera/same_as_mine.jpg", b),
            (
                "Camera/IMG_20260716_120000.jpg",
                jpeg(photo(5), exif=camera_exif("2026:07:16 12:00:00", model="SM-S911B")),
            ),
        ],
    )
    return {"mine": mine, "sam": sam}
