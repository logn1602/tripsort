# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=11.1"]
# ///
"""A small demo trip made only from public-domain and CC0 images, to show and check tripsort.

    uv run tools/demo_trip.py build OUT          # OUT/{lisa,george,pearl}.zip, OUT/refs/, truth
    tripsort run OUT/lisa.zip OUT/george.zip OUT/pearl.zip -o SORTED --refs OUT/refs
    uv run tools/demo_trip.py check SORTED OUT   # the run's output against OUT/truth.json

Every image comes from tests/fixtures (sources and licenses in tests/fixtures/README.md). Three
public-domain portrait paintings play three friends, Lisa (the Mona Lisa), George (Gilbert
Stuart's George Washington), and Pearl (Girl with a Pearl Earring): on their own, pasted in
front of the sample's views and rooms, and side by side in group shots. The rest of the trip is
the public category sample, all 25 photos, with the lion out of focus as the photo to flag.
tripsort puts 24 of the 25 in the right category: the road bike lands in animals, a known miss
(the README's known limits) that ``check`` allows and the GIF marks. Like a real trip, the
three zips share some photos byte for byte, hold WhatsApp copies (smaller, recompressed, no
EXIF) of others, a three-shot burst where only the middle shot is sharp, and a little junk.

``truth.json`` says what each zip member is. ``check`` reads the run's manifest.json and
folders and fails (exit 1) unless tripsort found what the trip holds: every copy, the three
friends named from refs/ with no wrong photo (and at most one missed), the categories (the
road bike in animals or in other, and at most one more photo wrong, since a close call may land
either way on another CPU), the burst and its best shot, and the blurry photo. The
fresh-install CI job, tests/test_demo.py, and tools/make_demo_gif.py use it. With the same
Pillow, every build writes the same bytes.
"""

import argparse
import io
import json
import sys
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from PIL import ExifTags, Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ZIPS = ("lisa", "george", "pearl")
FRIENDS = {
    "lisa": "mona_lisa",
    "george": "george_washington",
    "pearl": "girl_with_a_pearl_earring",
}
CATEGORIES = ("scenery", "animals", "food", "documents")
TRUTH = "truth.json"
_EXIF_TIME = "%Y:%m:%d %H:%M:%S"
_ZIP_TIME = (2026, 7, 16, 21, 0, 0)  # every member's zip timestamp, so the zips never change


@dataclass
class Member:
    """One zip member and what tripsort should make of it."""

    zip: str
    member: str
    role: str  # photo, exact_duplicate, copy, junk
    origin: str | None = None  # the photo it shows (copies and duplicates: their original's)
    people: list[str] = field(default_factory=list)
    category: str | None = None  # scenery, animals, food, documents, or other
    shot: str | None = None  # solo, group, or none
    burst: str | None = None
    burst_best: bool = False
    reject: str | None = None
    known_miss: str | None = None  # the category tripsort puts it in instead (a known limit)


@dataclass
class Photo:
    origin: str
    owner: str  # the zip of the friend whose phone took it
    image: Callable[[], Image.Image]
    people: tuple[str, ...] = ()
    category: str = "other"
    burst: str | None = None
    burst_best: bool = False
    reject: str | None = None
    seconds_after_previous: int | None = None  # burst shots follow the shot before them
    known_miss: str | None = None


def fixture(relative: str) -> Image.Image:
    with Image.open(FIXTURES / relative) as image:
        return image.convert("RGB")


def painting(friend: str) -> Image.Image:
    return fixture(f"paintings/{FRIENDS[friend]}.jpg")


def _bust(friend: str, mirrored: bool) -> Image.Image:
    """Head and shoulders: the top three quarters of the painting."""
    image = painting(friend)
    image = image.crop((0, 0, image.width, int(image.height * 0.75)))
    return ImageOps.mirror(image) if mirrored else image


def _soft_mask(size: tuple[int, int]) -> Image.Image:
    """Opaque inside, fading out over the outer 6%: no hard painted frame in the scene."""
    w, h = size
    edge = max(2, int(min(w, h) * 0.06))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (edge, edge, w - edge, h - edge), radius=edge * 2, fill=255
    )
    return mask.filter(ImageFilter.GaussianBlur(edge / 2))


def with_friends(background: str, friends: list[tuple[str, bool]]) -> Callable[[], Image.Image]:
    """Friends (name, mirrored) standing in front of a fixture photo, left to right."""

    def make() -> Image.Image:
        scene = fixture(background)
        share = {1: 0.86, 2: 0.78, 3: 0.68}[len(friends)]
        slots = {1: (0.58,), 2: (0.06, 0.52), 3: (0.02, 0.35, 0.68)}[len(friends)]
        for (friend, mirrored), left in zip(friends, slots, strict=True):
            bust = _bust(friend, mirrored)
            height = int(scene.height * share)
            bust = bust.resize(
                (bust.width * height // bust.height, height), Image.Resampling.LANCZOS
            )
            x = min(int(scene.width * left), scene.width - bust.width)
            scene.paste(bust, (x, scene.height - bust.height), _soft_mask(bust.size))
        return scene

    return make


def _shifted(relative: str, dx: float, blur: float) -> Callable[[], Image.Image]:
    """Another shot of a burst: the camera moved a little, maybe out of focus."""

    def make() -> Image.Image:
        image = fixture(relative)
        w, h = image.size
        cut = int(w * abs(dx))
        box = (cut, 0, w, h) if dx > 0 else (0, 0, w - cut, h)
        image = image.crop(box).resize((w, h), Image.Resampling.BICUBIC)
        return image.filter(ImageFilter.GaussianBlur(blur)) if blur else image

    return make


def _plain(relative: str) -> Callable[[], Image.Image]:
    return lambda: fixture(relative)


def trip() -> list[Photo]:
    """Every photo of the trip, in the order they were taken."""
    lisa, george, pearl = ZIPS
    return [
        Photo("lisa_portrait", lisa, lambda: painting("lisa"), ("lisa",)),
        Photo("beach", lisa, _plain("categories/scenery/beach_driftwood.jpg"), category="scenery"),
        Photo(
            "lisa_pearl_beach",
            george,
            with_friends(
                "categories/scenery/beach_driftwood.jpg", [("lisa", False), ("pearl", True)]
            ),
            ("lisa", "pearl"),
        ),
        Photo(
            "pearl_beach",
            lisa,
            with_friends("categories/scenery/beach_driftwood.jpg", [("pearl", True)]),
            ("pearl",),
        ),
        Photo("dog", lisa, _plain("categories/animals/dog_on_pier.jpg"), category="animals"),
        Photo("kingfisher", pearl, _plain("categories/animals/kingfisher.jpg"), category="animals"),
        Photo("curry", pearl, _plain("categories/food/curry_rice.jpg"), category="food"),
        Photo("risotto", lisa, _plain("categories/food/prawn_risotto.jpg"), category="food"),
        Photo(
            "pearl_market",
            george,
            with_friends("categories/food/produce_display.jpg", [("pearl", False)]),
            ("pearl",),
        ),
        Photo(
            "ticket", george, _plain("categories/documents/train_ticket.jpg"), category="documents"
        ),
        Photo("george_portrait", pearl, lambda: painting("george"), ("george",)),
        Photo("river", george, _plain("categories/scenery/mountain_river.jpg"), category="scenery"),
        Photo(
            "george_river",
            lisa,
            with_friends("categories/scenery/mountain_river.jpg", [("george", True)]),
            ("george",),
        ),
        Photo(
            "lisa_river",
            pearl,
            with_friends("categories/scenery/mountain_river.jpg", [("lisa", False)]),
            ("lisa",),
        ),
        Photo(
            "lion_blurry",
            george,
            lambda: fixture("categories/animals/lion.jpg").filter(ImageFilter.GaussianBlur(3)),
            category="animals",
            reject="blurry",
        ),
        Photo("flamingo", pearl, _plain("categories/animals/flamingo.jpg"), category="animals"),
        Photo("map", lisa, _plain("categories/documents/city_map_1855.jpg"), category="documents"),
        Photo("pearl_portrait", george, lambda: painting("pearl"), ("pearl",)),
        Photo(
            "george_lisa_city",
            pearl,
            with_friends(
                "categories/scenery/city_skyline.jpg", [("george", False), ("lisa", True)]
            ),
            ("george", "lisa"),
        ),
        Photo("city", lisa, _plain("categories/scenery/city_skyline.jpg"), category="scenery"),
        Photo(
            "pearl_city",
            george,
            with_friends("categories/scenery/city_skyline.jpg", [("pearl", False)]),
            ("pearl",),
        ),
        Photo("oysters", george, _plain("categories/food/oysters_lemon.jpg"), category="food"),
        Photo("dishes", pearl, _plain("categories/food/dishes_on_table.jpg"), category="food"),
        Photo("produce", lisa, _plain("categories/food/produce_display.jpg"), category="food"),
        Photo(
            "theatre_ticket",
            pearl,
            _plain("categories/documents/theatre_ticket.jpg"),
            category="documents",
        ),
        Photo(
            "receipt",
            george,
            _plain("categories/documents/grocery_receipt.jpg"),
            category="documents",
        ),
        Photo(
            "alert", lisa, _plain("categories/documents/alert_screenshot.jpg"), category="documents"
        ),
        Photo("hotel", pearl, _plain("categories/other/hotel_room.jpg")),
        Photo(
            "george_hotel",
            pearl,
            with_friends("categories/other/hotel_room.jpg", [("george", False)]),
            ("george",),
        ),
        Photo("cat", george, _plain("categories/animals/cat_on_bench.jpg"), category="animals"),
        Photo("tools", george, _plain("categories/other/tool_kit.jpg")),
        Photo("car", lisa, _plain("categories/other/vintage_car.jpg")),
        Photo(
            "lisa_car",
            george,
            with_friends("categories/other/vintage_car.jpg", [("lisa", True)]),
            ("lisa",),
        ),
        Photo("chairs", pearl, _plain("categories/other/empty_chairs.jpg")),
        Photo(
            "george_chairs",
            lisa,
            with_friends("categories/other/empty_chairs.jpg", [("george", True)]),
            ("george",),
        ),
        Photo(
            "sunset_tree", pearl, _plain("categories/scenery/sunset_tree.jpg"), category="scenery"
        ),
        Photo(
            "all_three",
            george,
            with_friends(
                "categories/scenery/sunset_tree.jpg",
                [("george", False), ("lisa", False), ("pearl", True)],
            ),
            ("george", "lisa", "pearl"),
        ),
        # A burst: the camera moves a little between shots, and only the middle one is sharp.
        Photo(
            "sunset_1",
            george,
            _shifted("categories/scenery/sunset_boats.jpg", -0.015, 0.8),
            category="scenery",
            burst="sunset",
        ),
        Photo(
            "sunset_2",
            george,
            _plain("categories/scenery/sunset_boats.jpg"),
            category="scenery",
            burst="sunset",
            burst_best=True,
            seconds_after_previous=1,
        ),
        Photo(
            "sunset_3",
            george,
            _shifted("categories/scenery/sunset_boats.jpg", 0.015, 0.8),
            category="scenery",
            burst="sunset",
            seconds_after_previous=1,
        ),
        # A photo of a thing, which tripsort puts in animals: the public sample's known miss (#22).
        Photo("road_bike", pearl, _plain("categories/other/road_bike.jpg"), known_miss="animals"),
    ]


# Photos a friend's zip holds byte for byte as well: (origin, zip).
EXACT_DUPLICATES = (
    ("dog", "pearl"),
    ("curry", "lisa"),
    ("pearl_city", "pearl"),
    ("ticket", "lisa"),
    ("river", "pearl"),
)
# WhatsApp copies: (origin, zip). Group shots are the ones that get shared.
WHATSAPP_COPIES = (
    ("lisa_pearl_beach", "lisa"),
    ("george_lisa_city", "george"),
    ("all_three", "pearl"),
    ("pearl_market", "lisa"),
    ("beach", "george"),
)
JUNK = (("lisa", ".DS_Store"), ("pearl", "__MACOSX/._20260714_101500.jpg"))


def _exif(taken: datetime, owner: str) -> bytes:
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Demo"
    exif[ExifTags.Base.Model] = f"Phone {owner[0].upper()}"
    exif[ExifTags.Base.DateTime] = taken.strftime(_EXIF_TIME)  # the modify date, as cameras do
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = taken.strftime(_EXIF_TIME)
    return exif.tobytes()


def _jpeg(image: Image.Image, quality: int, exif: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    if exif is None:
        image.save(buf, "JPEG", quality=quality)
    else:
        image.save(buf, "JPEG", quality=quality, exif=exif)
    return buf.getvalue()


def _camera_name(owner: str, taken: datetime, counter: int) -> str:
    """File names the way each friend's phone writes them."""
    if owner == "lisa":
        return f"IMG_{4100 + counter:04d}.jpg"
    if owner == "george":
        return f"PXL_{taken:%Y%m%d_%H%M%S}{counter % 1000:03d}.jpg"
    return f"{taken:%Y%m%d_%H%M%S}.jpg"


def build(out: Path) -> list[Member]:
    """Write the three zips, refs/, and truth.json into ``out``; return the members."""
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, list[tuple[str, bytes]]] = {name: [] for name in ZIPS}
    members: list[Member] = []
    data_of: dict[str, bytes] = {}
    image_of: dict[str, Image.Image] = {}
    photo_of: dict[str, Photo] = {}
    taken_of: dict[str, datetime] = {}
    taken = datetime(2026, 7, 14, 9, 4, 0)
    for counter, photo in enumerate(trip()):
        if photo.seconds_after_previous is not None:
            taken += timedelta(seconds=photo.seconds_after_previous)
        else:
            taken += timedelta(minutes=29 + (counter * 7) % 23)
        image = photo.image()
        data = _jpeg(image, 90, _exif(taken, photo.owner))
        member = _camera_name(photo.owner, taken, counter)
        entries[photo.owner].append((member, data))
        data_of[photo.origin], image_of[photo.origin] = data, image
        photo_of[photo.origin], taken_of[photo.origin] = photo, taken
        members.append(_member(photo.owner, member, "photo", photo))

    for origin, zip_name in EXACT_DUPLICATES:
        photo = photo_of[origin]
        member = _camera_name(zip_name, taken_of[origin], len(entries[zip_name]) + 500)
        entries[zip_name].append((member, data_of[origin]))
        members.append(_member(zip_name, member, "exact_duplicate", photo))
    for k, (origin, zip_name) in enumerate(WHATSAPP_COPIES):
        image = image_of[origin].copy()
        image.thumbnail((480, 480), Image.Resampling.LANCZOS)
        member = f"WhatsApp Images/IMG-{taken_of[origin]:%Y%m%d}-WA{k + 1:04d}.jpg"
        entries[zip_name].append((member, _jpeg(image, 70)))
        members.append(_member(zip_name, member, "copy", photo_of[origin]))
    for zip_name, member in JUNK:
        entries[zip_name].append((member, b"\x00\x05\x16\x07junk"))
        members.append(Member(zip_name, member, "junk"))

    for name in ZIPS:
        with zipfile.ZipFile(out / f"{name}.zip", "w") as archive:
            for member, data in sorted(entries[name]):
                info = zipfile.ZipInfo(member, date_time=_ZIP_TIME)
                archive.writestr(info, data)

    refs = out / "refs"
    refs.mkdir(exist_ok=True)
    for friend in FRIENDS:  # a brighter copy of each painting, as a separate photo would be
        bright = ImageEnhance.Brightness(painting(friend)).enhance(1.3)
        (refs / f"{friend}.jpg").write_bytes(_jpeg(bright, 90))
    (out / TRUTH).write_text(
        json.dumps([asdict(m) for m in members], indent=1) + "\n", encoding="utf-8"
    )
    return members


def _member(zip_name: str, member: str, role: str, photo: Photo) -> Member:
    shot = "none" if not photo.people else "solo" if len(photo.people) == 1 else "group"
    return Member(
        zip=zip_name,
        member=member,
        role=role,
        origin=photo.origin,
        people=sorted(photo.people),
        category=photo.category,
        shot=shot,
        burst=photo.burst,
        burst_best=photo.burst_best,
        reject=photo.reject,
        known_miss=photo.known_miss,
    )


def check(sorted_dir: Path, trip_dir: Path) -> list[str]:
    """Problems with a run's output (``tripsort run`` on the trip in ``trip_dir``, written to
    ``sorted_dir``); an empty list when tripsort found what the trip holds."""
    truth = [Member(**m) for m in json.loads((trip_dir / TRUTH).read_text(encoding="utf-8"))]
    manifest = json.loads((sorted_dir / "manifest.json").read_text(encoding="utf-8"))
    by_member = {(m.zip, m.member): m for m in truth}
    problems: list[str] = []

    kept: dict[str, dict[str, object]] = {}  # origin -> its kept file in the manifest
    for file in manifest["files"]:
        origins = {
            by_member[(o["source"], o["member"])].origin
            for o in file["occurrences"]
            if (o["source"], o["member"]) in by_member
        }
        if len(origins) != 1:
            problems.append(f"one file holds {len(origins)} photos: {sorted(map(str, origins))}")
            continue
        (origin,) = origins
        if file["keeper"]:
            if origin in kept:
                problems.append(f"{origin}: kept twice (a copy was missed)")
            else:
                kept[str(origin)] = file
        elif file["copy_of"] is None:
            problems.append(f"{origin}: neither kept nor a copy ({file['status']})")
    originals = {m.origin: m for m in truth if m.role == "photo"}
    problems += [f"{o}: not kept" for o in sorted(set(originals) - set(kept))]

    summary = manifest["summary"]
    duplicates = sum(m.role == "exact_duplicate" for m in truth)
    copies = sum(m.role == "copy" for m in truth)
    if summary["exact_duplicates"] != duplicates:
        problems.append(f"{summary['exact_duplicates']} exact duplicates, expected {duplicates}")
    if summary["near_copies"] != copies:
        problems.append(f"{summary['near_copies']} copies, expected {copies}")

    named = {p["name"]: p["named_by"] for p in manifest["people"] if p["named_by"]}
    if named != dict.fromkeys(FRIENDS, "refs"):
        problems.append(f"named people {sorted(named)}, expected {sorted(FRIENDS)} from refs/")
    for friend in FRIENDS:
        expected = {o for o, m in originals.items() if friend in m.people}
        found = {o for o, f in kept.items() if friend in f["people"]}  # type: ignore[operator]
        wrong, missed = found - expected, expected - found
        if wrong:
            problems.append(f"{friend}: wrong photos {sorted(wrong)}")
        if len(missed) > 1:
            problems.append(f"{friend}: missed {sorted(missed)}")
        folder = sorted_dir / "people" / friend
        if len(list(folder.glob("*"))) != len(found):
            problems.append(f"people/{friend}/ does not hold the {len(found)} photos found")

    wrong_categories = []  # a photo in the category of its known miss does not count
    for o, m in originals.items():
        got = str(kept[o]["category"] or "other") if o in kept else m.category
        if got not in (m.category, m.known_miss):
            wrong_categories.append(f"{o} ({m.category} as {got})")
    if len(wrong_categories) > 1:
        problems.append(f"wrong categories: {wrong_categories}")
    for name in CATEGORIES:
        if not list((sorted_dir / "categories" / name).glob("*")):
            problems.append(f"categories/{name}/ is empty")

    burst_shots = {o for o, m in originals.items() if m.burst}
    groups = {kept[o]["burst"] for o in burst_shots if o in kept}
    in_burst = {o for o, f in kept.items() if f["burst"] is not None}
    if len(groups) != 1 or None in groups or in_burst != burst_shots:
        problems.append(f"burst: {sorted(in_burst)} grouped, expected {sorted(burst_shots)}")
    best = {o for o in burst_shots if o in kept and kept[o]["burst_best"]}
    expected_best = {o for o in burst_shots if originals[o].burst_best}
    if best != expected_best:
        problems.append(f"burst: best shot {sorted(best)}, expected {sorted(expected_best)}")

    rejects = {o: str(f["reject"]) for o, f in kept.items() if f["reject"]}
    expected_rejects = {o: str(m.reject) for o, m in originals.items() if m.reject}
    if rejects != expected_rejects:
        problems.append(f"rejects {rejects}, expected {expected_rejects}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the demo zips, refs/, and truth.json")
    b.add_argument("out", type=Path)
    c = sub.add_parser("check", help="check a tripsort run of the demo trip")
    c.add_argument("sorted", type=Path, help="the folder tripsort run wrote (-o)")
    c.add_argument("trip", type=Path, help="the folder build wrote")
    args = parser.parse_args(argv)
    if args.command == "build":
        members = build(args.out)
        photos = sum(m.role != "junk" for m in members)
        print(f"demo trip: {photos} photos in {len(ZIPS)} zips, written to {args.out}")
        return 0
    if not (args.sorted / "manifest.json").is_file():
        print(f"demo check FAILED: no manifest.json in {args.sorted}", file=sys.stderr)
        return 1
    problems = check(args.sorted, args.trip)
    for problem in problems:
        print(f"demo check: {problem}", file=sys.stderr)
    if problems:
        print(f"demo check FAILED: {len(problems)} problems", file=sys.stderr)
        return 1
    print("demo check passed: tripsort sorted the demo trip as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
