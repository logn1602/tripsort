"""Read the EXIF fields tripsort uses. Broken EXIF counts as no EXIF, never as an error."""

import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PIL import ExifTags, Image

_DATETIME = re.compile(r"^\s*(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_SUBSEC = re.compile(r"^\s*(\d+)")
_OFFSET = re.compile(r"^\s*([+-])(\d{2}):(\d{2})\s*$")
_ROTATED = frozenset({5, 6, 7, 8})


@dataclass(frozen=True)
class ExifInfo:
    datetime: str | None = None  # 'YYYY-MM-DDTHH:MM:SS' plus '.fff' when sub-seconds exist
    offset: str | None = None  # '+HH:MM'
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    orientation: int = 1

    @property
    def has_camera_exif(self) -> bool:
        """A capture time or a camera make or model, not just an orientation tag."""
        return self.datetime is not None or self.make is not None or self.model is not None

    @property
    def swaps_axes(self) -> bool:
        """True when the orientation turns the stored pixels by 90 degrees."""
        return self.orientation in _ROTATED


NO_EXIF = ExifInfo()


def read_exif(image: Image.Image) -> ExifInfo:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Pillow warns about every corrupt EXIF block
            return _read(image.getexif())
    except Exception:  # any malformed EXIF block means "no EXIF"
        return NO_EXIF


def _read(exif: Image.Exif) -> ExifInfo:
    ifd = exif.get_ifd(ExifTags.IFD.Exif)
    stamp = parse_exif_datetime(
        ifd.get(ExifTags.Base.DateTimeOriginal), ifd.get(ExifTags.Base.SubsecTimeOriginal)
    )
    offset = parse_offset(ifd.get(ExifTags.Base.OffsetTimeOriginal))
    if stamp is None:
        stamp = parse_exif_datetime(
            ifd.get(ExifTags.Base.DateTimeDigitized), ifd.get(ExifTags.Base.SubsecTimeDigitized)
        )
        offset = parse_offset(ifd.get(ExifTags.Base.OffsetTimeDigitized))
    if stamp is None:
        offset = None
    lat, lon = _gps(exif.get_ifd(ExifTags.IFD.GPSInfo))
    orientation = exif.get(ExifTags.Base.Orientation, 1)
    return ExifInfo(
        datetime=stamp,
        offset=offset,
        make=_text(exif.get(ExifTags.Base.Make)),
        model=_text(exif.get(ExifTags.Base.Model)),
        lens=_text(ifd.get(ExifTags.Base.LensModel)),
        gps_lat=lat,
        gps_lon=lon,
        orientation=orientation if isinstance(orientation, int) and 1 <= orientation <= 8 else 1,
    )


def parse_exif_datetime(value: Any, subsec: Any = None) -> str | None:
    """'2026:07:14 10:23:45' plus SubSecTime '123' -> '2026-07-14T10:23:45.123'."""
    text = _text(value)
    if text is None:
        return None
    match = _DATETIME.match(text)
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(part) for part in match.groups())
    try:
        stamp = datetime(year, month, day, hour, minute, second)
    except ValueError:  # e.g. '0000:00:00 00:00:00'
        return None
    if not 1900 <= stamp.year <= 2100:
        return None
    iso = stamp.isoformat()
    sub = _text(subsec)
    if sub is not None and (digits := _SUBSEC.match(sub)) is not None:
        iso += "." + digits.group(1)[:6]
    return iso


def parse_offset(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    match = _OFFSET.match(text)
    if match is None:
        return None
    sign, hours, minutes = match.groups()
    if int(hours) > 14 or int(minutes) > 59:
        return None
    return f"{sign}{hours}:{minutes}"


def _text(value: Any) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    cleaned = value.replace("\x00", "").strip()
    return cleaned or None


def _gps(gps: dict[int, Any]) -> tuple[float | None, float | None]:
    lat = _degrees(gps.get(ExifTags.GPS.GPSLatitude), gps.get(ExifTags.GPS.GPSLatitudeRef), 90)
    lon = _degrees(gps.get(ExifTags.GPS.GPSLongitude), gps.get(ExifTags.GPS.GPSLongitudeRef), 180)
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):  # 0,0 means "no fix"
        return None, None
    return lat, lon


def _degrees(value: Any, ref: Any, limit: float) -> float | None:
    try:
        degrees, minutes, seconds = (float(part) for part in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    result = degrees + minutes / 60 + seconds / 3600
    if not math.isfinite(result) or result > limit:
        return None
    if _text(ref) in {"S", "W"}:
        result = -result
    return round(result, 7)
