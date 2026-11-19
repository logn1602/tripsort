"""Decode an image once, at reduced size, and compute what later stages need from pixels.

Pillow's decompression-bomb check runs when an image is opened, before tripsort can use
JPEG draft mode, so a 200 MP phone photo would be refused even though draft mode decodes
it at 1/8 scale. Images of the supported formats are therefore opened through Pillow's
plugin registry without that check, and this module applies its own limit instead:
the pixels actually decoded must stay within ``PIL.Image.MAX_IMAGE_PIXELS``. A file over
the limit is not unreadable; it is passed through without a pHash, so it still takes part
in exact dedupe and export but not in the near-copy check.
"""

import io
import math
import warnings
from dataclasses import dataclass

import pi_heif
from PIL import Image, ImageOps

from tripsort.ingest.exif import NO_EXIF, ExifInfo, read_exif
from tripsort.ingest.hashing import phash_hex

DECODE_MAX_SIDE = 1600  # JPEGs are decoded in draft mode to about this size
PHASH_SOURCE_SIDE = 256  # shorter side, in pixels, the pHash input is reduced towards
_SNIFFED_FORMATS = ("JPEG", "PNG", "WEBP", "HEIF")
_DRAFT_FORMATS = frozenset({"JPEG", "MPO"})
_SAFE_MODES = frozenset({"1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK", "YCbCr"})
_plugins_ready = False


class UnreadableImageError(Exception):
    """The bytes could not be decoded as an image."""


@dataclass(frozen=True)
class ImageAnalysis:
    format: str | None
    width: int | None  # full size, after EXIF orientation
    height: int | None
    exif: ExifInfo
    phash: str | None
    decoded: bool  # whether pixels were decoded
    note: str | None = None


def _ensure_plugins() -> None:
    global _plugins_ready
    if not _plugins_ready:
        Image.init()
        pi_heif.register_heif_opener()
        _plugins_ready = True


def open_image(data: bytes) -> Image.Image:
    """Open lazily (no pixels decoded), skipping the bomb check for the sniffed formats."""
    _ensure_plugins()
    prefix = data[:16]
    for name in _SNIFFED_FORMATS:
        factory, accept = Image.OPEN[name]
        if accept is None or accept(prefix) is True:
            return factory(io.BytesIO(data), "")
    return Image.open(io.BytesIO(data))  # anything else Pillow knows, with its usual checks


def draft_box(size: tuple[int, int], max_side: int) -> tuple[int, int] | None:
    """A box with the image's aspect ratio and ``max_side`` on its long side.

    JPEG draft mode only shrinks when both sides of the request fit, so asking for a square
    box would decode a 4000x3000 photo at full size.
    """
    width, height = size
    long_side = max(width, height)
    if long_side <= max_side:
        return None
    scale = long_side / max_side
    return max(1, math.ceil(width / scale)), max(1, math.ceil(height / scale))


def analyze_image(data: bytes, *, max_side: int = DECODE_MAX_SIDE) -> ImageAnalysis:
    """Decode ``data`` at reduced size and return its metadata and pHash.

    Raises ``UnreadableImageError`` when the bytes are not a decodable image.
    """
    with warnings.catch_warnings():
        # Pillow warns about odd but harmless files (truncated EXIF, malformed MPO data);
        # one line per photo would bury the summary.
        warnings.simplefilter("ignore")
        return _analyze(data, max_side)


def _analyze(data: bytes, max_side: int) -> ImageAnalysis:
    try:
        image = open_image(data)
    except Image.DecompressionBombError as exc:
        return ImageAnalysis(None, None, None, NO_EXIF, None, False, f"not decoded: {exc}")
    except Exception as exc:  # decoders raise many exception types on bad input
        raise UnreadableImageError(_describe(exc)) from exc

    with image:
        fmt = image.format
        exif = read_exif(image)
        width, height = image.size
        if exif.swaps_axes:
            width, height = height, width

        if fmt in _DRAFT_FORMATS and (box := draft_box(image.size, max_side)) is not None:
            image.draft("RGB", box)
        decoded_pixels = image.size[0] * image.size[1]
        limit = Image.MAX_IMAGE_PIXELS
        if limit is not None and decoded_pixels > limit:
            note = (
                f"{width}x{height} is over the {limit / 1e6:.0f} MP decode limit; "
                "checked for exact duplicates only"
            )
            return ImageAnalysis(fmt, width, height, exif, None, False, note)

        try:
            image.load()
        except Exception as exc:  # same: a broken file can fail in many ways
            raise UnreadableImageError(_describe(exc)) from exc

        try:
            phash = phash_hex(_prepare(image))
        except Exception as exc:  # keep the file, just without a pHash
            return ImageAnalysis(
                fmt, width, height, exif, None, True, f"no pHash: {_describe(exc)}"
            )
    return ImageAnalysis(fmt, width, height, exif, phash, True)


def _prepare(image: Image.Image) -> Image.Image:
    """The image the pHash is computed from: shrunk, then turned by its EXIF orientation.

    The pHash only looks at a 32x32 version, so a fast integer reduction to a few hundred
    pixels comes first; resampling a 1600 px image straight to 32 px costs far more and
    changes nothing at the frequencies the pHash keeps.
    """
    if image.mode not in _SAFE_MODES:
        image = image.convert("L" if len(image.getbands()) == 1 else "RGB")
    factor = min(image.size) // PHASH_SOURCE_SIDE
    if factor > 1:
        image = image.reduce(factor)  # keeps .info, so the EXIF orientation still applies
    return ImageOps.exif_transpose(image)


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
