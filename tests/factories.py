"""Generated test data: photo-like images, EXIF blocks, HEIC files, and zips.

No real photos are used anywhere in the tests. HEIC fixtures are encoded through
pillow-heif's own API and never registered with Pillow, so decoding always goes through
pi-heif, the decoder tripsort ships with.
"""

import io
import struct
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import ExifTags, Image, ImageOps, TiffImagePlugin


def photo(seed: int, size: tuple[int, int] = (1200, 900)) -> Image.Image:
    """A smooth random RGB image; different seeds give very different pHashes."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)


def exif_block(
    *,
    datetime: str | None = None,
    subsec: str | None = None,
    offset: str | None = None,
    digitized: str | None = None,
    make: str | None = None,
    model: str | None = None,
    lens: str | None = None,
    orientation: int | None = None,
    gps: tuple[float, float] | None = None,
    exposure: tuple[int, int] | None = None,
    iso: int | tuple[int, ...] | None = None,
) -> bytes:
    exif = Image.Exif()
    if make is not None:
        exif[ExifTags.Base.Make] = make
    if model is not None:
        exif[ExifTags.Base.Model] = model
    if orientation is not None:
        exif[ExifTags.Base.Orientation] = orientation
    ifd = exif.get_ifd(ExifTags.IFD.Exif)
    if datetime is not None:
        ifd[ExifTags.Base.DateTimeOriginal] = datetime
    if subsec is not None:
        ifd[ExifTags.Base.SubsecTimeOriginal] = subsec
    if offset is not None:
        ifd[ExifTags.Base.OffsetTimeOriginal] = offset
    if digitized is not None:
        ifd[ExifTags.Base.DateTimeDigitized] = digitized
    if lens is not None:
        ifd[ExifTags.Base.LensModel] = lens
    if exposure is not None:
        ifd[ExifTags.Base.ExposureTime] = TiffImagePlugin.IFDRational(*exposure)
    if iso is not None:
        ifd[ExifTags.Base.ISOSpeedRatings] = iso
    if gps is not None:
        lat, lon = gps
        g = exif.get_ifd(ExifTags.IFD.GPSInfo)
        g[ExifTags.GPS.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        g[ExifTags.GPS.GPSLatitude] = _dms(abs(lat))
        g[ExifTags.GPS.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        g[ExifTags.GPS.GPSLongitude] = _dms(abs(lon))
    return exif.tobytes()


def _dms(value: float) -> tuple[float, float, float]:
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = round(((value - degrees) * 60 - minutes) * 60, 4)
    return float(degrees), float(minutes), seconds


def camera_exif(datetime: str = "2026:07:14 10:23:45", **extra: object) -> bytes:
    """EXIF the way a phone camera writes it."""
    values: dict[str, object] = {
        "make": "Google",
        "model": "Pixel 7",
        "datetime": datetime,
        "exposure": (1, 120),
        "iso": 100,
    }
    values.update(extra)
    return exif_block(**values)  # type: ignore[arg-type]


def encode(
    image: Image.Image, fmt: str = "JPEG", *, exif: bytes | None = None, **opts: object
) -> bytes:
    buf = io.BytesIO()
    if exif is not None:
        opts["exif"] = exif
    image.save(buf, fmt, **opts)
    return buf.getvalue()


def jpeg(image: Image.Image, *, quality: int = 90, exif: bytes | None = None) -> bytes:
    return encode(image, "JPEG", exif=exif, quality=quality)


def heic(image: Image.Image, *, exif: bytes | None = None, quality: int = 70) -> bytes:
    heif = pillow_heif.from_pillow(image)
    if exif is not None:
        heif.info["exif"] = exif
    buf = io.BytesIO()
    heif.save(buf, quality=quality)
    return buf.getvalue()


def whatsapp_copy(data: bytes, *, max_side: int = 600, quality: int = 70) -> bytes:
    """What a messenger does: apply the rotation, shrink, recompress, strip EXIF."""
    with Image.open(io.BytesIO(data)) as im:
        oriented = ImageOps.exif_transpose(im).convert("RGB")
    oriented.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return jpeg(oriented, quality=quality)


def make_zip(
    path: Path,
    entries: Iterable[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, data in entries:
            if name.endswith("/"):
                zf.mkdir(name.rstrip("/"))
            else:
                zf.writestr(name, data)
    return path


def patch_entry(
    data: bytes,
    name: str,
    *,
    new_name: bytes | None = None,
    set_flags: int = 0,
    method: int | None = None,
    corrupt: bool = False,
) -> bytes:
    """Rewrite one entry's headers in raw zip bytes (for cases zipfile cannot write).

    ``new_name`` must have the same length as ``name``; ``corrupt`` flips a data byte so
    the CRC no longer matches.
    """
    buf = bytearray(data)
    encoded = name.encode("ascii")
    eocd = buf.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack_from("<II", buf, eocd + 12)
    pos = cd_offset
    found = False
    while pos < cd_offset + cd_size:
        assert buf[pos : pos + 4] == b"PK\x01\x02"
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", buf, pos + 28)
        (local,) = struct.unpack_from("<I", buf, pos + 42)
        if bytes(buf[pos + 46 : pos + 46 + name_len]) == encoded:
            found = True
            for header, flags_at, method_at, name_at in ((pos, 8, 10, 46), (local, 6, 8, 30)):
                (flags,) = struct.unpack_from("<H", buf, header + flags_at)
                struct.pack_into("<H", buf, header + flags_at, flags | set_flags)
                if method is not None:
                    struct.pack_into("<H", buf, header + method_at, method)
                if new_name is not None:
                    assert len(new_name) == len(encoded)
                    buf[header + name_at : header + name_at + len(encoded)] = new_name
            if corrupt:
                local_name, local_extra = struct.unpack_from("<HH", buf, local + 26)
                buf[local + 30 + local_name + local_extra + 5] ^= 0xFF
        pos += 46 + name_len + extra_len + comment_len
    assert found, name
    return bytes(buf)
