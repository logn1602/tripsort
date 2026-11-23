"""Draw docs/demo.gif: the demo trip before and after ``tripsort run``.

    uv run tools/make_demo_gif.py [--out docs/demo.gif] [--keep DIR]

It builds the demo trip of tools/demo_trip.py (public-domain and CC0 images only) in a
temporary folder, runs ``tripsort run`` on it with this interpreter (the face and CLIP models
download on first use), checks the result with demo_trip.py's check, and draws every frame
from what the run produced: the zips' members, the summary tripsort printed, manifest.json,
and the exported folders. Nothing in it is typed in by hand. A photo in a category it does not
belong to (by the trip's truth.json) is marked wrong, not left out. ``--keep DIR`` keeps the
trip and the sorted folder there.

Fonts come from the system (Segoe UI and Consolas on Windows, where the committed GIF was
drawn; Helvetica and Menlo on macOS; DejaVu on Linux), else Pillow's own.
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_trip  # the sibling script

OUT = Path(__file__).resolve().parents[1] / "docs" / "demo.gif"
WIDTH, HEIGHT = 960, 540
COMMAND = ["tripsort", "run", "lisa.zip", "george.zip", "pearl.zip", "-o", "sorted/"]
COMMAND += ["--refs", "refs/"]

Color = tuple[int, int, int]
BACKGROUND: Color = (13, 17, 23)
PANEL: Color = (22, 27, 34)
CELL: Color = (33, 39, 47)
TERMINAL: Color = (8, 11, 16)
TEXT: Color = (230, 237, 243)
SOFT: Color = (201, 209, 217)
MUTED: Color = (139, 148, 158)
GUIDE: Color = (72, 79, 88)
ACCENT: Color = (88, 166, 255)
HIGHLIGHT: Color = (31, 58, 96)
RED: Color = (218, 54, 51)
GREEN: Color = (46, 160, 67)
LIGHTS: tuple[Color, ...] = ((255, 95, 86), (255, 189, 46), (39, 201, 63))
UI_COLORS = (BACKGROUND, PANEL, CELL, TERMINAL, TEXT, SOFT, MUTED, GUIDE, ACCENT, HIGHLIGHT)
UI_COLORS += (RED, GREEN, *LIGHTS)

SANS = ("segoeui.ttf", "HelveticaNeue.ttc", "Helvetica.ttc", "DejaVuSans.ttf")
SANS_BOLD = ("seguisb.ttf", "segoeuib.ttf", "HelveticaNeue.ttc", "DejaVuSans-Bold.ttf")
MONO = ("consola.ttf", "Menlo.ttc", "DejaVuSansMono.ttf")
MONO_BOLD = ("consolab.ttf", "Menlo.ttc", "DejaVuSansMono-Bold.ttf")

NOTES = {
    "all": "every photo once: copies removed, the best version kept, named by date",
    "shots/solo": "one friend in front of the camera",
    "shots/group": "two or more friends in front of the camera",
    "categories/scenery": "views where the view itself is the subject",
    "categories/animals": "animals",
    "categories/food": "food",
    "categories/documents": "tickets, maps, receipts, and screenshots",
    "bursts": "the shots of one moment side by side; the sharpest, best_, goes in the folders",
    "rejects/blurry": "flagged to check: still in every other folder, nothing deleted",
}
SKIPPED = {"shots/solo"}  # in the tree, but not shown on its own: the GIF stays short
LARGEST_CELL = 136  # px; bigger photos make a bigger file (each folder's frame redraws them)
HOLD_MS = {"before": 2400, "found": 2600, "typing": 300, "output": 3800, "all": 1700}
FOLDER_MS = 1150
MARKED_MS = 2400  # a folder with a photo marked wrong: time to read why


@dataclass
class Folder:
    path: str  # relative to the sorted folder, such as people/lisa
    files: list[Path]
    note: str
    best: Path | None = None  # in a burst's folder, its best shot
    wrong: list[Path] = field(default_factory=list)  # in a category's, photos of another one


@dataclass
class Row:
    """A line of the folder tree."""

    prefix: str
    name: str
    count: int | None = None
    folder: int | None = None  # index into Run.folders


@dataclass
class Run:
    zips: dict[str, list[tuple[str, bytes]]]  # zip -> its photos (member, bytes), in zip order
    found: dict[tuple[str, str], str]  # (zip, member) -> kept, duplicate, or copy
    summary: dict[str, int]
    output: list[str]  # what tripsort run printed
    folders: list[Folder]  # in the order the tree shows them
    tree: list[Row] = field(default_factory=list)
    palette: Image.Image | None = None


@dataclass
class Frame:
    image: Image.Image
    ms: int


@cache
def font(size: int, kind: str = "sans") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = {"sans": SANS, "bold": SANS_BOLD, "mono": MONO, "mono-bold": MONO_BOLD}[kind]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def load_run(trip_dir: Path, sorted_dir: Path, output: str) -> Run:
    """Everything the frames show, read from the trip and the run's output."""
    truth = json.loads((trip_dir / demo_trip.TRUTH).read_text(encoding="utf-8"))
    junk = {(m["zip"], m["member"]) for m in truth if m["role"] == "junk"}
    zips: dict[str, list[tuple[str, bytes]]] = {}
    for name in demo_trip.ZIPS:
        with zipfile.ZipFile(trip_dir / f"{name}.zip") as archive:
            zips[name] = [
                (info.filename, archive.read(info))
                for info in archive.infolist()
                if (name, info.filename) not in junk
            ]
    # tripsort exports a photo byte for byte, so its bytes tell which member of the trip it is.
    member_of = {(m["zip"], m["member"]): m for m in truth}
    truth_of = {data: member_of[(z, member)] for z in zips for member, data in zips[z]}

    manifest = json.loads((sorted_dir / "manifest.json").read_text(encoding="utf-8"))
    order = {name: i for i, name in enumerate(demo_trip.ZIPS)}
    found: dict[tuple[str, str], str] = {}
    for file in manifest["files"]:
        places = sorted(
            ((o["source"], o["member"]) for o in file["occurrences"]),
            key=lambda place: (order.get(place[0], len(order)), place[1]),
        )
        for k, place in enumerate(places):
            # The zip read first keeps a file; the same bytes in a later zip are a duplicate.
            found[place] = "copy" if not file["keeper"] else "kept" if k == 0 else "duplicate"

    def files_in(path: str) -> list[Path]:
        folder = sorted_dir / path
        return sorted(f for f in folder.iterdir() if f.is_file()) if folder.is_dir() else []

    def subfolders(path: str) -> list[str]:
        folder = sorted_dir / path
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    folders: list[Folder] = []
    tree: list[Row] = []

    def add(prefix: str, name: str, path: str, note: str, files: list[Path]) -> None:
        best = next((f for f in files if f.name.startswith("best_")), None)
        wrong: list[Path] = []
        if path.startswith("categories/"):  # what tripsort got wrong is marked, not left out
            category = path.removeprefix("categories/")
            wrong = [f for f in files if truth_of[f.read_bytes()]["category"] != category]
            members = [truth_of[f.read_bytes()] for f in wrong]
            if members:
                what = ", ".join(f"the {m['origin'].replace('_', ' ')}" for m in members)
                known = all(m.get("known_miss") == category for m in members)
                note = f"{note}; marked wrong: {what}{', a known limit' if known else ''}"
        tree.append(Row(prefix, name, len(files), len(folders)))
        folders.append(Folder(path, files, note, best, wrong))

    def branch(name: str, children: list[tuple[str, str, str]]) -> None:
        tree.append(Row("├─ ", f"{name}/"))
        for k, (label, path, note) in enumerate(children):
            prefix = "│  └─ " if k == len(children) - 1 else "│  ├─ "
            add(prefix, f"{label}/", path, note, files_in(path))

    add("├─ ", "all/", "all", NOTES["all"], files_in("all"))
    people = [p["name"] for p in manifest["people"] if p["photos"]]
    branch(
        "people",
        [
            (p, f"people/{p}", f"every photo {p.title()} is in, named from refs/{p}.jpg")
            for p in people
        ],
    )
    branch("shots", [(s, f"shots/{s}", NOTES[f"shots/{s}"]) for s in ("solo", "group")])
    branch(
        "categories",
        [(c, f"categories/{c}", NOTES[f"categories/{c}"]) for c in demo_trip.CATEGORIES],
    )
    bursts = subfolders("bursts")
    shots = [  # in the order they were taken: best_ goes in front of the best shot's name
        f
        for b in bursts
        for f in sorted(files_in(f"bursts/{b}"), key=lambda f: f.name.removeprefix("best_"))
    ]
    add("├─ ", "bursts/", "bursts", NOTES["bursts"], shots)
    tree[-1].count = len(bursts)
    branch(
        "rejects",
        [(r, f"rejects/{r}", NOTES.get(f"rejects/{r}", "")) for r in subfolders("rejects")],
    )
    tree.append(Row("└─ ", "manifest.json"))

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    return Run(zips, found, manifest["summary"], lines, folders, tree)


@cache
def _decoded(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.draft("RGB", (400, 400))  # JPEG: decode at a fraction of the size
        return image.convert("RGB")


def make_palette(run: Run) -> Image.Image:
    """One palette for every frame: the trip's photos' colors plus the exact colors of the
    interface, so text and panels never shift between frames."""
    photos = [data for members in run.zips.values() for _, data in members]
    mosaic = Image.new("RGB", (96 * len(photos), 96))
    for k, data in enumerate(photos):
        mosaic.paste(_decoded(data).resize((96, 96)), (96 * k, 0))
    colors = mosaic.quantize(256 - len(UI_COLORS), method=Image.Quantize.MEDIANCUT)
    flat = list(colors.getpalette() or [])[: 3 * (256 - len(UI_COLORS))]
    flat += [v for color in UI_COLORS for v in color]
    palette = Image.new("P", (1, 1))
    palette.putpalette(flat)
    return palette


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    return image, ImageDraw.Draw(image)


def paste_in_cell(
    run: Run, image: Image.Image, data: bytes, box: tuple[int, int, int, int], dim: bool = False
) -> None:
    """A photo, fit into a cell. It is dithered to the palette here, before badges go on it,
    so the same photo in the same cell is the same pixels in every frame."""
    x0, y0, x1, y1 = box
    ImageDraw.Draw(image).rounded_rectangle(box, radius=4, fill=CELL)
    photo = _decoded(data).copy()
    photo.thumbnail((x1 - x0 - 4, y1 - y0 - 4), Image.Resampling.LANCZOS)
    if dim:
        photo = Image.blend(photo, Image.new("RGB", photo.size, BACKGROUND), 0.6)
    if run.palette is not None:
        photo = photo.quantize(palette=run.palette, dither=Image.Dither.FLOYDSTEINBERG)
        photo = photo.convert("RGB")
    image.paste(photo, (x0 + (x1 - x0 - photo.width) // 2, y0 + (y1 - y0 - photo.height) // 2))


def badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: Color) -> None:
    small = font(12, "bold")
    x, y = xy
    width = int(draw.textlength(text, font=small)) + 10
    draw.rounded_rectangle((x, y, x + width, y + 18), radius=4, fill=color)
    draw.text((x + 5, y + 1), text, font=small, fill=TEXT)


def fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    face: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    width: int,
) -> str:
    """``text``, cut short with an ellipsis where it would run past ``width``."""
    if draw.textlength(text, font=face) <= width:
        return text
    while text and draw.textlength(text + "…", font=face) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def right_aligned(draw: ImageDraw.ImageDraw, right: int, y: int, text: str, size: int) -> None:
    draw.text(
        (right - draw.textlength(text, font=font(size)), y), text, font=font(size), fill=MUTED
    )


def before(run: Run, marked: bool) -> Image.Image:
    image, draw = canvas()
    total = sum(len(m) for m in run.zips.values())
    draw.text((36, 22), "Before: everyone's zips from the trip", font=font(28, "bold"), fill=TEXT)
    subtitle = f"{total} photos from three phones, with the same shots shared back and forth"
    draw.text((36, 62), subtitle, font=font(17), fill=MUTED)
    panel_w, top = 280, 104
    gap = (WIDTH - 2 * 36 - 3 * panel_w) // 2
    for i, name in enumerate(demo_trip.ZIPS):
        x = 36 + i * (panel_w + gap)
        draw.rounded_rectangle((x, top, x + panel_w, top + 372), radius=8, fill=PANEL)
        draw.text((x + 16, top + 12), f"{name}.zip", font=font(19, "mono"), fill=TEXT)
        right_aligned(draw, x + panel_w - 16, top + 14, f"{len(run.zips[name])} photos", 16)
        for k, (member, data) in enumerate(run.zips[name]):
            cx, cy = x + 12 + (k % 4) * 65, top + 50 + (k // 4) * 62
            status = run.found.get((name, member), "kept")
            dropped = marked and status != "kept"
            paste_in_cell(run, image, data, (cx, cy, cx + 61, cy + 58), dim=dropped)
            if dropped:
                badge(draw, (cx + 3, cy + 37), "copy" if status == "copy" else "dup", RED)
    if marked:
        s = run.summary
        text = (
            f"{s['exact_duplicates']} exact duplicates and {s['near_copies']} WhatsApp copies "
            f"found: {s['kept']} photos kept, each in its best version"
        )
        draw.text((36, 492), text, font=font(18, "bold"), fill=ACCENT)
    return image


def terminal(typed: str, output: Sequence[str]) -> Image.Image:
    image, draw = canvas()
    draw.rounded_rectangle((24, 20, WIDTH - 24, HEIGHT - 20), radius=10, fill=TERMINAL)
    for k, color in enumerate(LIGHTS):
        draw.ellipse((44 + k * 22, 36, 56 + k * 22, 48), fill=color)
    mono = font(16, "mono")
    draw.text((48, 70), "$", font=mono, fill=GREEN)
    draw.text((66, 70), typed, font=mono, fill=TEXT)
    for k, line in enumerate(output):
        draw.text((48, 106 + k * 24), fit(draw, line, mono, WIDTH - 96), font=mono, fill=MUTED)
    return image


def grid_layout(n: int, width: int, height: int, largest: int = LARGEST_CELL) -> tuple[int, int]:
    """(columns, cell side) that fits n square cells, 6 px apart, in width x height, with
    the cells as big as they can be (at most ``largest``)."""
    best = (1, 0)
    for columns in range(1, max(n, 1) + 1):
        rows = -(-n // columns)
        side = min(largest, (width + 6) // columns - 6, (height + 6) // max(rows, 1) - 6)
        if side >= best[1]:  # on a tie, more columns: fewer rows
            best = (columns, side)
    return best


def after(run: Run, shown: int) -> Image.Image:
    image, draw = canvas()
    mono, mono_bold = font(16, "mono"), font(16, "mono-bold")
    draw.rounded_rectangle((24, 20, 332, HEIGHT - 20), radius=8, fill=PANEL)
    draw.text((42, 32), "sorted/", font=font(18, "mono"), fill=TEXT)
    for k, row in enumerate(run.tree):
        y = 62 + k * 24
        here = row.folder == shown
        if here:
            draw.rounded_rectangle((34, y - 3, 322, y + 20), radius=4, fill=HIGHLIGHT)
        color = TEXT if here else MUTED if row.folder is None else SOFT
        draw.text((42, y), row.prefix, font=mono, fill=GUIDE)
        x = 42 + draw.textlength(row.prefix, font=mono)
        draw.text((x, y), row.name, font=mono_bold if here else mono, fill=color)
        if row.count is not None:
            label = str(row.count)
            draw.text((310 - draw.textlength(label, font=mono), y), label, font=mono, fill=color)

    folder = run.folders[shown]
    x0, x1 = 348, WIDTH - 24
    draw.rounded_rectangle((x0, 20, x1, HEIGHT - 20), radius=8, fill=PANEL)
    draw.text((x0 + 18, 32), f"{folder.path}/", font=font(22, "mono"), fill=TEXT)
    n = len(folder.files)
    right_aligned(draw, x1 - 18, 36, f"{n} photo{'' if n == 1 else 's'}", 17)
    note = fit(draw, folder.note, font(16), x1 - x0 - 36)
    draw.text((x0 + 18, 68), note, font=font(16), fill=MUTED)
    columns, side = grid_layout(n, x1 - x0 - 36, HEIGHT - 20 - 108 - 16)
    for k, path in enumerate(folder.files):
        cx = x0 + 18 + (k % columns) * (side + 6)
        cy = 108 + (k // columns) * (side + 6)
        paste_in_cell(run, image, path.read_bytes(), (cx, cy, cx + side, cy + side))
        if path == folder.best:
            badge(draw, (cx + 6, cy + 6), "best", GREEN)
        if path in folder.wrong:
            badge(draw, (cx + 6, cy + 6), "wrong", RED)
    return image


def frames(run: Run) -> list[Frame]:
    if run.palette is None:
        run.palette = make_palette(run)
    out = [
        Frame(before(run, marked=False), HOLD_MS["before"]),
        Frame(before(run, marked=True), HOLD_MS["found"]),
    ]
    for words in (2, 5, 7, 9):  # the command, typed a few words at a time
        out.append(Frame(terminal(" ".join(COMMAND[:words]), []), HOLD_MS["typing"]))
    out.append(Frame(terminal(" ".join(COMMAND), run.output[:16]), HOLD_MS["output"]))
    for k, folder in enumerate(run.folders):
        if folder.path not in SKIPPED:
            hold = HOLD_MS.get(folder.path, MARKED_MS if folder.wrong else FOLDER_MS)
            out.append(Frame(after(run, k), hold))
    return out


def save_gif(sequence: Sequence[Frame], palette: Image.Image, path: Path) -> None:
    """Every frame on the one palette without dithering (the photos already are), so what
    does not change between frames is the same pixels and the GIF stores only the changes."""
    indexed = [f.image.quantize(palette=palette, dither=Image.Dither.NONE) for f in sequence]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=[f.ms for f in sequence],
        loop=0,
        disposal=1,
    )


def run_demo(work: Path) -> tuple[Path, Path, str]:
    """Build the trip in ``work``, run tripsort on it, check the output; return the trip
    folder, the sorted folder, and what tripsort printed."""
    trip = work / "trip"
    demo_trip.build(trip)
    command = [sys.executable, "-m", "tripsort", *COMMAND[1:]]
    result = subprocess.run(command, cwd=trip, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"tripsort run failed ({result.returncode}):\n{result.stderr}")
    sorted_dir = trip / "sorted"
    problems = demo_trip.check(sorted_dir, trip)
    if problems:
        raise SystemExit("the demo run is not as expected:\n" + "\n".join(problems))
    return trip, sorted_dir, result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT, help=f"default: {OUT}")
    parser.add_argument("--keep", type=Path, help="keep the trip and the sorted folder here")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="tripsort-demo-") as tmp:
        trip, sorted_dir, output = run_demo(Path(tmp))
        run = load_run(trip, sorted_dir, output)
        sequence = frames(run)
        assert run.palette is not None
        save_gif(sequence, run.palette, args.out)
        if args.keep is not None:
            shutil.copytree(trip, args.keep, dirs_exist_ok=True)
    seconds = sum(f.ms for f in sequence) / 1000
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}: {len(sequence)} frames, {seconds:.1f} s, {size:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
