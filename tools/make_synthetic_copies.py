"""Make zips of known copies and hard negatives, to measure dedupe precision and recall.

    uv run tools/make_synthetic_copies.py OUT_DIR --generate 200
    uv run tools/make_synthetic_copies.py OUT_DIR --from PHOTO_DIR [--limit 400]

Writes three files to OUT_DIR (keep it outside the repository):

* originals.zip: the originals, unchanged, plus hard negatives that must NOT be merged
* copies.zip: recompressed copies of each original, as messengers and cloud apps make them
* truth.csv: which original every member came from (zip, member, origin, role, variant)

Members are named by synthetic ids, never by the source file names. Score the result with
``uv run eval/run_eval.py dedupe OUT_DIR``.

Copies (role "copy"):
  whatsapp       rotation applied, long side 1600 px, JPEG quality 75, EXIF stripped
  whatsapp_hd    rotation applied, long side 4096 px, quality 85, EXIF stripped
  telegram       rotation applied, long side 1280 px, quality 82, EXIF stripped
  storage_saver  long side 2048 px, quality 85, EXIF kept (Google Photos storage saver)
  recompress     full size, quality 60, EXIF kept (re-saved by an editor)
  heic_to_jpeg   HEIC originals only: full size JPEG, EXIF kept (iPhone "Automatic" transfer)

Hard negatives (role "negative", each its own origin, all in originals.zip):
  burst_exif         a burst sibling (1% camera shift, new noise), EXIF time one second later
  burst_same_second  two burst siblings with the same EXIF second, no sub-seconds, and the
                     same exposure and ISO (a locked burst: no EXIF field tells them apart)
  burst_same_second_exposure_differs
                     the same, but the exposure time and ISO differ between the two shots
  burst_no_exif      two burst siblings without EXIF (pHash alone cannot separate these)
  document           phone screenshots of one ticket layout with different text
"""

import argparse
import csv
import hashlib
import io
import random
import string
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pillow_heif  # encoding only; never registered with Pillow
from PIL import ExifTags, Image, ImageDraw, ImageOps, TiffImagePlugin

from tripsort.ingest.decode import open_image
from tripsort.ingest.zip_reader import IMAGE_EXTENSIONS

COPY_VARIANTS = (
    "whatsapp",
    "whatsapp_hd",
    "telegram",
    "storage_saver",
    "recompress",
    "heic_to_jpeg",
)
NEGATIVE_FAMILIES = (
    "burst_exif",
    "burst_same_second",
    "burst_same_second_exposure_differs",
    "burst_no_exif",
    "document",
)
EXPOSURES = ((1, 60), (1, 120), (1, 250), (1, 500))
ISOS = (50, 100, 200, 400)
_EXIF_TIME = "%Y:%m:%d %H:%M:%S"


@dataclass
class Row:
    zip: str
    member: str
    origin: str
    role: str  # original, copy, negative
    variant: str


@dataclass
class Original:
    data: bytes  # the file exactly as it was
    extension: str
    image: Image.Image  # decoded, stored orientation (EXIF rotation not applied)
    exif: bytes | None
    is_heic: bool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="output folder (outside the repository)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from", dest="photos", type=Path, help="folder of photos to copy")
    source.add_argument("--generate", type=int, metavar="N", help="generate N photo-like images")
    parser.add_argument("--limit", type=int, help="use at most this many photos from --from")
    parser.add_argument("--size", default="4000x3000", help="generated size (default 4000x3000)")
    parser.add_argument("--documents", type=int, help="document screenshots (default N/5, min 4)")
    parser.add_argument(
        "--negatives-every",
        type=int,
        default=3,
        metavar="K",
        help="make burst negatives from every K-th original (default 3)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.photos is not None:
        originals = list(_from_folder(args.photos, args.limit, rng))
    else:
        width, height = (int(v) for v in args.size.lower().split("x"))
        originals = list(_generated(args.generate, (width, height), rng))
    if not originals:
        print("no usable photos found", file=sys.stderr)
        return 1
    documents = args.documents if args.documents is not None else max(4, len(originals) // 5)

    rows: list[Row] = []
    with (
        zipfile.ZipFile(args.out / "originals.zip", "w") as originals_zip,
        zipfile.ZipFile(args.out / "copies.zip", "w") as copies_zip,
    ):
        for i, original in enumerate(originals):
            origin = f"o{i:05d}"
            member = f"originals/{origin}{original.extension}"
            originals_zip.writestr(member, original.data)
            rows.append(Row("originals", member, origin, "original", "original"))
            for variant in COPY_VARIANTS:
                data = _copy(original, variant)
                if data is not None:
                    member = f"copies/{origin}_{variant}.jpg"
                    copies_zip.writestr(member, data)
                    rows.append(Row("copies", member, origin, "copy", variant))
            if i % args.negatives_every == 0:
                for family, data in _bursts(original, rng):
                    n = sum(1 for r in rows if r.role == "negative")
                    negative = f"n{n:05d}"
                    member = f"negatives/{negative}_{family}.jpg"
                    originals_zip.writestr(member, data)
                    rows.append(Row("originals", member, negative, "negative", family))
            original.image.close()
        for k, data in enumerate(_documents(documents, rng)):
            member = f"negatives/d{k:05d}_document.png"
            originals_zip.writestr(member, data)
            rows.append(Row("originals", member, f"d{k:05d}", "negative", "document"))

    with (args.out / "truth.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["zip", "member", "origin", "role", "variant"])
        writer.writerows([r.zip, r.member, r.origin, r.role, r.variant] for r in rows)
    roles = {
        role: sum(1 for r in rows if r.role == role) for role in ("original", "copy", "negative")
    }
    print(f"{roles['original']} originals, {roles['copy']} copies, {roles['negative']} negatives")
    return 0


def _from_folder(folder: Path, limit: int | None, rng: random.Random) -> Iterator[Original]:
    paths = sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None and len(paths) > limit:
        paths = sorted(rng.sample(paths, limit))
    seen: set[str] = set()
    for path in paths:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:  # an exact duplicate is not an interesting test case
            continue
        seen.add(digest)
        try:
            image = open_image(data)
            image.load()
        except Exception:  # unreadable source photos are simply not used
            continue
        exif = image.info.get("exif")
        is_heic = image.format == "HEIF"
        yield Original(
            data,
            ".heic" if is_heic else ".jpg" if image.format == "JPEG" else path.suffix.lower(),
            image,
            exif if isinstance(exif, bytes) else None,
            is_heic,
        )


def _generated(n: int, size: tuple[int, int], rng: random.Random) -> Iterator[Original]:
    start = datetime(2026, 7, 14, 9, 0, 0)
    for i in range(n):
        image = _photo_like(rng.randrange(2**32), size)
        taken = start + timedelta(minutes=7 * i, seconds=rng.randrange(60))
        exif = Image.Exif()
        exif[ExifTags.Base.Make] = "Synthetic"
        exif[ExifTags.Base.Model] = f"Camera {i % 3}"
        ifd = exif.get_ifd(ExifTags.IFD.Exif)
        ifd[ExifTags.Base.DateTimeOriginal] = taken.strftime(_EXIF_TIME)
        ifd[ExifTags.Base.SubsecTimeOriginal] = f"{rng.randrange(1000):03d}"
        ifd[ExifTags.Base.ExposureTime] = TiffImagePlugin.IFDRational(*rng.choice(EXPOSURES))
        ifd[ExifTags.Base.ISOSpeedRatings] = rng.choice(ISOS)
        if i % 5 == 4:  # some phones store pixels sideways and rotate with EXIF
            exif[ExifTags.Base.Orientation] = 6
            image = image.transpose(Image.Transpose.ROTATE_90)
        raw = exif.tobytes()
        if i % 7 == 6:
            heif = pillow_heif.from_pillow(image)
            heif.info["exif"] = raw
            buf = io.BytesIO()
            heif.save(buf, quality=80)
            decoded = open_image(buf.getvalue())
            decoded.load()
            yield Original(buf.getvalue(), ".heic", decoded, decoded.info.get("exif"), True)
        else:
            data = _jpeg(image, 92, raw)
            decoded = open_image(data)  # decoded from the bytes, so its EXIF rotation is known
            decoded.load()
            yield Original(data, ".jpg", decoded, raw, False)


def _photo_like(seed: int, size: tuple[int, int]) -> Image.Image:
    """Smooth multi-scale noise: random, but with photo-like low-frequency structure."""
    rng = np.random.default_rng(seed)
    width, height = size
    small = (max(1, width // 8), max(1, height // 8))
    acc = np.zeros((small[1], small[0], 3), dtype=np.float32)
    for cells, weight in ((4, 1.0), (16, 0.5), (64, 0.25), (256, 0.12)):
        grid = rng.random((max(2, cells * height // width), cells, 3), dtype=np.float32)
        layer = Image.fromarray((grid * 255).astype(np.uint8)).resize(
            small, Image.Resampling.BICUBIC
        )
        acc += weight * np.asarray(layer, dtype=np.float32)
    acc = acc / acc.max() * 255
    return Image.fromarray(acc.astype(np.uint8)).resize(size, Image.Resampling.BICUBIC)


def _jpeg(image: Image.Image, quality: int, exif: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    options: dict[str, object] = {"quality": quality}
    if exif:
        options["exif"] = exif
    image.convert("RGB").save(buf, "JPEG", **options)
    return buf.getvalue()


def _shrunk(image: Image.Image, long_side: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return copy


def _copy(original: Original, variant: str) -> bytes | None:
    image, exif = original.image, original.exif
    if variant in {"whatsapp", "whatsapp_hd", "telegram"}:
        side, quality = {"whatsapp": (1600, 75), "whatsapp_hd": (4096, 85), "telegram": (1280, 82)}[
            variant
        ]
        return _jpeg(
            ImageOps.exif_transpose(_shrunk(image, side)), quality
        )  # shrink first: less memory
    if variant == "storage_saver":
        return _jpeg(_shrunk(image, 2048), 85, exif)
    if variant == "recompress":
        return _jpeg(image, 60, exif)
    if variant == "heic_to_jpeg":
        return _jpeg(image, 92, exif) if original.is_heic else None
    raise ValueError(variant)


def _sibling(image: Image.Image, rng: random.Random) -> Image.Image:
    """Another shot of the same moment: the camera moved about 1%, new sensor noise."""
    oriented = ImageOps.exif_transpose(_shrunk(image, 2400)).convert("RGB")  # 2400 px: less memory
    width, height = oriented.size
    dx = max(1, int(width * rng.uniform(0.006, 0.012)))
    dy = max(1, int(height * rng.uniform(0.004, 0.010)))
    shifted = oriented.crop((dx, dy, width, height)).resize(
        (width, height), Image.Resampling.BICUBIC
    )
    noise_rng = np.random.default_rng(rng.randrange(2**32))
    noise = noise_rng.standard_normal((height, width, 3), dtype=np.float32) * 4
    pixels = np.clip(np.asarray(shifted, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def _burst_exif(
    base: str,
    offset_seconds: int,
    *,
    subsec: str | None,
    model: str,
    exposure: tuple[int, int],
    iso: int,
) -> bytes:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Synthetic"
    exif[ExifTags.Base.Model] = model
    stamp = datetime.strptime(base, _EXIF_TIME) + timedelta(seconds=offset_seconds)
    ifd = exif.get_ifd(ExifTags.IFD.Exif)
    ifd[ExifTags.Base.DateTimeOriginal] = stamp.strftime(_EXIF_TIME)
    if subsec is not None:
        ifd[ExifTags.Base.SubsecTimeOriginal] = subsec
    ifd[ExifTags.Base.ExposureTime] = TiffImagePlugin.IFDRational(*exposure)
    ifd[ExifTags.Base.ISOSpeedRatings] = iso
    return exif.tobytes()


def _settings(ifd: dict[int, object]) -> tuple[tuple[int, int], int]:
    """The original's exposure time and ISO, or typical values when it records none."""
    exposure: tuple[int, int] = (1, 120)
    value = ifd.get(ExifTags.Base.ExposureTime)
    if (
        isinstance(value, TiffImagePlugin.IFDRational)
        and value.numerator > 0
        and value.denominator > 0
    ):
        exposure = (int(value.numerator), int(value.denominator))
    raw_iso = ifd.get(ExifTags.Base.ISOSpeedRatings)
    if isinstance(raw_iso, tuple) and raw_iso:
        raw_iso = raw_iso[0]
    iso = int(raw_iso) if isinstance(raw_iso, int) and raw_iso > 0 else 100
    return exposure, iso


def _bursts(original: Original, rng: random.Random) -> Iterator[tuple[str, bytes]]:
    """Burst siblings of one original. The EXIF families need a capture time on the original;
    without one they would test the no-EXIF rule instead."""
    stamp, model = None, "Burst camera"
    exposure, iso = (1, 120), 100
    if original.exif:
        info = Image.Exif()
        info.load(original.exif)
        ifd = info.get_ifd(ExifTags.IFD.Exif)
        found = ifd.get(ExifTags.Base.DateTimeOriginal)
        if isinstance(found, str) and len(found) >= 19:
            try:
                datetime.strptime(found[:19], _EXIF_TIME)
            except ValueError:
                pass
            else:
                stamp = found[:19]
        model = str(info.get(ExifTags.Base.Model) or model)
        exposure, iso = _settings(ifd)
    if stamp is not None:
        for offset, subsec in ((1, "250"), (2, "500")):  # later shots of the same burst
            exif = _burst_exif(
                stamp, offset, subsec=subsec, model=model, exposure=exposure, iso=iso
            )
            yield "burst_exif", _jpeg(_sibling(original.image, rng), 92, exif)
        for _ in range(2):  # two shots in one second; no sub-seconds; exposure locked
            exif = _burst_exif(stamp, 30, subsec=None, model=model, exposure=exposure, iso=iso)
            yield "burst_same_second", _jpeg(_sibling(original.image, rng), 92, exif)
        brighter = (exposure[0], max(1, exposure[1] // 2))  # twice as long
        for shot_exposure, shot_iso in ((exposure, iso), (brighter, iso * 2)):
            exif = _burst_exif(
                stamp, 40, subsec=None, model=model, exposure=shot_exposure, iso=shot_iso
            )
            yield (
                "burst_same_second_exposure_differs",
                _jpeg(_sibling(original.image, rng), 92, exif),
            )
    for _ in range(2):
        yield "burst_no_exif", _jpeg(_sibling(original.image, rng), 92)


def _documents(n: int, rng: random.Random) -> Iterator[bytes]:
    """Phone screenshots of one boarding-pass layout: the same structure, different text."""
    for _ in range(n):
        image = Image.new("RGB", (1080, 2400), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1080, 260), fill=(20, 60, 140))
        draw.text(
            (60, 90), "".join(rng.choices(string.ascii_uppercase, k=12)), fill="white", font_size=90
        )
        for line in range(14):
            words = [
                "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(3, 9)))
                for _ in range(4)
            ]
            draw.text((60, 380 + line * 120), " ".join(words), fill=(30, 30, 30), font_size=48)
        for col in range(6):
            for row in range(6):
                if rng.random() < 0.5:
                    x, y = 640 + col * 60, 2050 + row * 50
                    draw.rectangle((x, y, x + 50, y + 45), fill="black")
        buf = io.BytesIO()
        image.save(buf, "PNG")
        yield buf.getvalue()


if __name__ == "__main__":
    sys.exit(main())
