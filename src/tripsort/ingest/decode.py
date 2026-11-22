"""Decode an image once, at reduced size, and compute what later stages need from pixels:
the metadata, the pHash, (when a face analyzer is given) the faces, and (when asked) the
CLIP input, from the same oriented pixels the faces are found in.

Pillow's decompression-bomb check runs when an image is opened, before tripsort can use
JPEG draft mode, so a 200 MP phone photo would be refused even though draft mode decodes
it at 1/8 scale. Images of the supported formats are therefore opened through Pillow's
plugin registry without that check, and this module applies its own limit instead:
the pixels actually decoded must stay within ``PIL.Image.MAX_IMAGE_PIXELS``. A file over
the limit is not unreadable; it is passed through without a pHash, so it still takes part
in exact dedupe and export but not in the near-copy check.
"""

import contextlib
import io
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pi_heif
from PIL import Image, ImageOps

from tripsort.clip.preprocess import clip_input
from tripsort.clip.types import ClipPixels
from tripsort.cull.quality import Quality, measure
from tripsort.faces.types import FaceAnalyzer, FaceScan, RGBImage
from tripsort.ingest.exif import NO_EXIF, ExifInfo, read_exif
from tripsort.ingest.hashing import phash_hex

DECODE_MAX_SIDE = 1600  # JPEGs are decoded in draft mode to about this size
PHASH_SOURCE_SIDE = 256  # shorter side, in pixels, the pHash input is reduced towards
FACE_SOURCE_MAX_SIDE = 2048  # other formats decode at full size; faces need no more than this
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
    faces: FaceScan | None = None  # None when no face analyzer was given
    clip: ClipPixels | None = None  # None unless the CLIP input was asked for
    quality: Quality | None = None  # sharpness and exposure; None when not decoded


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


def analyze_image(
    data: bytes,
    *,
    max_side: int = DECODE_MAX_SIDE,
    faces: FaceAnalyzer | None = None,
    clip: bool = False,
) -> ImageAnalysis:
    """Decode ``data`` at reduced size and return its metadata, pHash, faces, and (with
    ``clip``) the CLIP input.

    Raises ``UnreadableImageError`` when the bytes are not a decodable image.
    """
    with warnings.catch_warnings():
        # Pillow warns about odd but harmless files (truncated EXIF, malformed MPO data);
        # one line per photo would bury the summary.
        warnings.simplefilter("ignore")
        return _analyze(data, max_side, faces, clip)


def _analyze(data: bytes, max_side: int, faces: FaceAnalyzer | None, clip: bool) -> ImageAnalysis:
    try:
        image = open_image(data)
    except Image.DecompressionBombError as exc:
        note = f"not decoded: {exc}"
        scan = FaceScan("not_decoded", detail=note) if faces is not None else None
        pixels = ClipPixels("not_decoded", detail=note) if clip else None
        return ImageAnalysis(None, None, None, NO_EXIF, None, False, note, scan, pixels)
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
            scan = FaceScan("not_decoded", detail=note) if faces is not None else None
            pixels = ClipPixels("not_decoded", detail=note) if clip else None
            return ImageAnalysis(fmt, width, height, exif, None, False, note, scan, pixels)

        try:
            image.load()
        except Exception as exc:  # same: a broken file can fail in many ways
            raise UnreadableImageError(_describe(exc)) from exc

        phash: str | None = None
        phash_note: str | None = None
        try:
            phash = phash_hex(_prepare(image))
        except Exception as exc:  # keep the file, just without a pHash
            phash_note = f"no pHash: {_describe(exc)}"
        quality: Quality | None = None
        with contextlib.suppress(Exception):  # nor do missing scores stop anything
            quality = measure(image)
        scan, pixels = _pixel_features(image, faces, clip)
    return ImageAnalysis(fmt, width, height, exif, phash, True, phash_note, scan, pixels, quality)


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


def face_source(image: Image.Image) -> RGBImage:
    """The decoded image as EXIF-oriented RGB pixels, at most about 2048 px on its long side.

    JPEGs arrive already shrunk by draft mode; other formats are box-reduced here first.
    """
    rgb = image if image.mode == "RGB" else image.convert("RGB")  # convert keeps .info
    factor = math.ceil(max(rgb.size) / FACE_SOURCE_MAX_SIDE)
    if factor > 1:
        rgb = rgb.reduce(factor)  # keeps .info, so the EXIF orientation still applies
    return np.asarray(ImageOps.exif_transpose(rgb))


def decode_oriented(data: bytes, max_side: int = DECODE_MAX_SIDE) -> RGBImage:
    """Decode ``data`` the way ingest does and return the pixels faces are found in.

    Raises ``UnreadableImageError`` when the bytes are not a decodable image.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            image = open_image(data)
            with image:
                if image.format in _DRAFT_FORMATS and (box := draft_box(image.size, max_side)):
                    image.draft("RGB", box)
                image.load()
                return face_source(image)
        except Exception as exc:  # decoders raise many exception types on bad input
            raise UnreadableImageError(_describe(exc)) from exc


def _pixel_features(
    image: Image.Image, analyzer: FaceAnalyzer | None, clip: bool
) -> tuple[FaceScan | None, ClipPixels | None]:
    """The face scan and the CLIP input, both from one EXIF-oriented RGB copy."""
    if analyzer is None and not clip:
        return None, None
    try:
        rgb = face_source(image)
    except Exception as exc:  # e.g. a pixel mode with no RGB conversion
        detail = f"could not prepare the pixels: {_describe(exc)}"
        return (
            FaceScan("failed", detail=detail) if analyzer is not None else None,
            ClipPixels("failed", detail=detail) if clip else None,
        )
    scan: FaceScan | None = None
    if analyzer is not None:
        try:
            scan = FaceScan("ok", analyzer.analyze(rgb))
        except Exception as exc:  # a face scan failure never stops the ingest
            scan = FaceScan("failed", detail=_describe(exc))
    pixels: ClipPixels | None = None
    if clip:
        try:
            pixels = ClipPixels("ok", clip_input(rgb))
        except Exception as exc:  # nor does a CLIP input failure
            pixels = ClipPixels("failed", detail=_describe(exc))
    return scan, pixels


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
