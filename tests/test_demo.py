"""The demo trip (tools/demo_trip.py) and the README's demo GIF (tools/make_demo_gif.py)."""

import json
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from tools import demo_trip, make_demo_gif
from tripsort.cli import app


@pytest.fixture(scope="module")
def trip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("demo") / "trip"
    demo_trip.build(folder)
    return folder


def perfect_run(trip: Path, out: Path, known_misses: bool = False) -> dict[str, Any]:
    """What `tripsort run` writes for the trip when it gets everything right (or everything
    but its known misses): manifest.json (the fields the checker and the GIF read) and the
    folders, with the photos copied out of the zips. Returns the manifest."""
    truth = [demo_trip.Member(**m) for m in json.loads((trip / "truth.json").read_text("utf-8"))]
    data: dict[tuple[str, str], bytes] = {}
    for name in demo_trip.ZIPS:
        with zipfile.ZipFile(trip / f"{name}.zip") as archive:
            data.update(((name, i.filename), archive.read(i)) for i in archive.infolist())
    files: list[dict[str, Any]] = []
    kept: dict[str, dict[str, Any]] = {}
    for m in truth:
        if m.role == "junk":
            continue
        place = {"source": m.zip, "member": m.member}
        if m.role == "copy":
            files.append({"occurrences": [place], "keeper": False, "copy_of": "sha", "people": []})
            continue
        category = (known_misses and m.known_miss) or m.category
        if m.origin not in kept:
            kept[str(m.origin)] = {
                "occurrences": [],
                "keeper": True,
                "copy_of": None,
                "status": "ok",
                "people": m.people,
                "category": None if category == "other" else category,
                "burst": 1 if m.burst else None,
                "burst_best": m.burst_best if m.burst else None,
                "reject": m.reject,
                "bytes": data[(m.zip, m.member)],
            }
            files.append(kept[str(m.origin)])
        kept[str(m.origin)]["occurrences"].append(place)

    def put(folder: str, name: str, content: bytes) -> None:
        (out / folder).mkdir(parents=True, exist_ok=True)
        (out / folder / name).write_bytes(content)

    photos: Counter[str] = Counter()
    for origin, file in kept.items():
        content = file.pop("bytes")
        name = f"20260714_{origin}.jpg"
        put("all", name, content)
        for person in file["people"]:
            put(f"people/{person}", name, content)
            photos[person] += 1
        if file["people"]:
            put(f"shots/{'solo' if len(file['people']) == 1 else 'group'}", name, content)
        if file["category"]:
            put(f"categories/{file['category']}", name, content)
        if file["burst"]:
            put("bursts/20260715_sunset_1", ("best_" if file["burst_best"] else "") + name, content)
        if file["reject"]:
            put(f"rejects/{file['reject']}", name, content)
    manifest = {
        "files": files,
        "people": [{"name": p, "photos": n, "named_by": "refs"} for p, n in photos.most_common()],
        "summary": {"exact_duplicates": 5, "near_copies": 5, "kept": len(kept)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_the_demo_trip_is_the_same_bytes_every_time(trip: Path, tmp_path: Path) -> None:
    members = demo_trip.build(tmp_path / "again")
    for name in [f"{z}.zip" for z in demo_trip.ZIPS] + ["truth.json", "refs/lisa.jpg"]:
        assert (trip / name).read_bytes() == (tmp_path / "again" / name).read_bytes(), name
    assert Counter(m.role for m in members) == {
        "photo": 41,
        "exact_duplicate": 5,
        "copy": 5,
        "junk": 2,
    }
    assert sorted(p.name for p in (trip / "refs").iterdir()) == [
        "george.jpg",
        "lisa.jpg",
        "pearl.jpg",
    ]
    originals = [m for m in members if m.role == "photo"]
    for friend in demo_trip.FRIENDS:  # enough faces for a person of their own
        assert sum(friend in m.people for m in originals) >= 5
    for category in demo_trip.CATEGORIES:
        assert sum(m.category == category for m in originals) >= 4
    assert [m.origin for m in originals if m.reject] == ["lion_blurry"]
    assert [m.origin for m in originals if m.burst_best] == ["sunset_2"]
    assert [(m.origin, m.known_miss) for m in originals if m.known_miss] == [
        ("road_bike", "animals")
    ]
    alone = [m for m in originals if not m.people and (m.burst is None or m.burst_best)]
    assert len(alone) == 25  # the whole public category sample, the burst counted once


def test_the_check_passes_a_run_that_got_everything_right(trip: Path, tmp_path: Path) -> None:
    perfect_run(trip, tmp_path / "sorted")
    assert demo_trip.check(tmp_path / "sorted", trip) == []
    assert demo_trip.main(["check", str(tmp_path / "sorted"), str(trip)]) == 0


def test_the_check_fails_without_a_manifest(
    trip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo_trip.main(["check", str(tmp_path / "never_run"), str(trip)]) == 1
    assert "no manifest.json" in capsys.readouterr().err


def _photo(trip: Path, manifest: dict[str, Any], origin: str) -> dict[str, Any]:
    """The manifest's kept file for one photo of the trip."""
    truth = json.loads((trip / "truth.json").read_text("utf-8"))
    member = next(m["member"] for m in truth if m["origin"] == origin and m["role"] == "photo")
    return next(
        f
        for f in manifest["files"]
        if f["keeper"] and any(o["member"] == member for o in f["occurrences"])
    )


def _wrong_person(manifest: dict[str, Any]) -> None:
    first = next(f for f in manifest["files"] if f["people"] == ["lisa"])
    first["people"] = ["george", "lisa"]


def _missed_copy(manifest: dict[str, Any]) -> None:
    copy = next(f for f in manifest["files"] if not f["keeper"])
    copy["keeper"], copy["copy_of"] = True, None


def _two_wrong_categories(manifest: dict[str, Any]) -> None:
    for f in [f for f in manifest["files"] if f.get("category") == "food"][:2]:
        f["category"] = "animals"


def _no_burst(manifest: dict[str, Any]) -> None:
    for f in manifest["files"]:
        if f.get("burst"):
            f["burst"], f["burst_best"] = None, None


def _extra_reject(manifest: dict[str, Any]) -> None:
    next(f for f in manifest["files"] if f["keeper"] and not f["reject"])["reject"] = "dark"


def _not_named(manifest: dict[str, Any]) -> None:
    manifest["people"][0]["named_by"] = None


@pytest.mark.parametrize(
    ("mistake", "problem"),
    [
        (_wrong_person, "george: wrong photos"),
        (_missed_copy, "kept twice"),
        (_two_wrong_categories, "wrong categories"),
        (_no_burst, "burst:"),
        (_extra_reject, "rejects"),
        (_not_named, "named people"),
    ],
)
def test_the_check_names_what_went_wrong(
    trip: Path,
    tmp_path: Path,
    mistake: Callable[[dict[str, Any]], None],
    problem: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "sorted"
    manifest = perfect_run(trip, out)
    mistake(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    problems = demo_trip.check(out, trip)
    assert any(problem in p for p in problems), problems
    assert demo_trip.main(["check", str(out), str(trip)]) == 1
    assert "demo check FAILED" in capsys.readouterr().err


def test_the_known_miss_and_one_close_call_are_tolerated(trip: Path, tmp_path: Path) -> None:
    """tripsort puts the road bike in animals (the README's known limits), and a close call
    may land either way on another CPU (see tests/test_categories.py)."""
    out = tmp_path / "sorted"
    manifest = perfect_run(trip, out, known_misses=True)
    assert _photo(trip, manifest, "road_bike")["category"] == "animals"
    _photo(trip, manifest, "lisa_portrait")["category"] = "documents"  # an `other` photo
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert demo_trip.check(out, trip) == []

    _photo(trip, manifest, "road_bike")["category"] = "food"  # not its known miss: it counts
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert any("wrong categories" in p for p in demo_trip.check(out, trip))


def test_the_gif_draws_the_run(trip: Path, tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    perfect_run(trip, out, known_misses=True)
    run = make_demo_gif.load_run(trip, out, "51 files in 3 zips\nKeeping 41 files\n")
    assert Counter(run.found.values()) == {"kept": 41, "duplicate": 5, "copy": 5}
    assert [f.path for f in run.folders][:4] == [
        "all",
        "people/lisa",
        "people/pearl",
        "people/george",
    ]
    assert run.folders[-2].path == "bursts"
    burst = run.folders[-2]
    assert burst.best is not None
    assert burst.files[1] == burst.best  # in the order taken: the middle shot is the best
    # The known miss is shown where tripsort put it, and marked.
    animals = next(f for f in run.folders if f.path == "categories/animals")
    assert [f.path for f in run.folders if f.wrong] == ["categories/animals"]
    assert len(animals.wrong) == 1
    assert animals.note.endswith("; marked wrong: the road bike, a known limit")

    frames = make_demo_gif.frames(run)
    assert run.palette is not None
    shown = len(run.folders) - len(make_demo_gif.SKIPPED)
    assert len(frames) == 2 + 4 + 1 + shown  # before, typing, the output, a frame per folder
    gif = tmp_path / "demo.gif"
    make_demo_gif.save_gif(frames, run.palette, gif)
    with Image.open(gif) as image:
        assert image.size == (make_demo_gif.WIDTH, make_demo_gif.HEIGHT)
        assert getattr(image, "n_frames", 1) == len(frames)
        assert image.info["loop"] == 0


def test_grid_layout_fills_the_area() -> None:
    assert make_demo_gif.grid_layout(40, 576, 400) == (8, 66)
    assert make_demo_gif.grid_layout(3, 576, 400, largest=176) == (3, 176)  # one row on a tie
    assert make_demo_gif.grid_layout(1, 576, 400) == (1, make_demo_gif.LARGEST_CELL)


@pytest.mark.models
@pytest.mark.cull  # the packaged reject thresholds: the blurry lion must be flagged
def test_tripsort_sorts_the_demo_trip(trip: Path, tmp_path: Path) -> None:
    """The real models on the demo trip, as the fresh-install CI job runs it."""
    out = tmp_path / "sorted"
    zips = [str(trip / f"{name}.zip") for name in demo_trip.ZIPS]
    result = CliRunner().invoke(app, ["run", *zips, "-o", str(out), "--refs", str(trip / "refs")])
    assert result.exit_code == 0, result.output
    assert demo_trip.check(out, trip) == []
