"""Culling: sharpness and exposure scores, rejects, and the best shot of each burst.

The tests that look at rejects are marked `cull`: elsewhere the flat generated photos would
all count as blurry, so conftest.py turns rejects off.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from factories import camera_exif, jpeg, make_zip
from fake_clip import HUES, ColorImageEncoder, WordTextEncoder
from fake_faces import MarkerFaceAnalyzer, face_photo
from PIL import Image, ImageFilter

from tripsort.api import Project, _category_lines
from tripsort.clip.bursts import BurstStats
from tripsort.clip.categorize import CategoryStats
from tripsort.config import load_config
from tripsort.cull.analysis import CullStats, _Scores, best_shot, rank_shots, reject_reason
from tripsort.cull.quality import measure
from tripsort.export.writer import ExportResult, _Burst, _own_best_shots

# the fake CLIP puts a photo on its color's axis, and these prompts name the axes
FAKE_CATEGORIES = """
[categories.scenery]
prompts = ["scenery"]
[categories.animals]
prompts = ["animals"]
[categories.food]
prompts = ["food"]
[categories.documents]
prompts = ["documents"]
[categories.other]
prompts = ["anything"]
"""


def textured(kind: str, seed: int, size: tuple[int, int] = (800, 600)) -> Image.Image:
    """A photo of one kind, in its HUES color so the fake CLIP sorts it, with fine texture."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-40, 41, size=(size[1], size[0], 1), dtype=np.int16)
    color = np.array(HUES[kind][0], dtype=np.int16)
    return Image.fromarray(np.clip(color + noise, 0, 255).astype(np.uint8))


def flat(level: int) -> Image.Image:
    return Image.new("RGB", (800, 600), (level, level, level))


def scores(**changes: float) -> _Scores:
    base = {
        "sharpness": 500.0,
        "sharp_tiles": 800.0,
        "luma_mean": 120.0,
        "dark_frac": 0.02,
        "bright_frac": 0.02,
    }
    return _Scores(**{**base, **changes})


def names(folder: Path) -> set[str]:
    return {p.name.rsplit("_", 1)[-1] for p in folder.iterdir()} if folder.is_dir() else set()


def test_blur_lowers_sharpness_and_exposure_finds_black_and_white_frames() -> None:
    sharp = textured("scenery", 1)
    assert measure(sharp.filter(ImageFilter.GaussianBlur(6))).sharp_tiles < (
        measure(sharp).sharp_tiles / 20
    )
    black, white = measure(flat(3)), measure(flat(252))
    assert (black.dark_frac, black.bright_frac) == (1.0, 0.0)
    assert black.luma_mean < 5
    assert (white.dark_frac, white.bright_frac) == (0.0, 1.0)
    # a soft background does not hide a sharp subject: the sharpest tiles still see it
    portrait = sharp.filter(ImageFilter.GaussianBlur(6))
    portrait.paste(sharp.crop((300, 200, 500, 400)), (300, 200))
    q = measure(portrait)
    assert q.sharp_tiles > 3 * q.sharpness


@pytest.mark.cull
def test_reject_rules_are_strict_and_say_why() -> None:
    t = load_config().thresholds
    assert reject_reason(scores(), t) is None
    assert reject_reason(scores(sharp_tiles=5.0), t) == "blurry"
    assert reject_reason(scores(sharp_tiles=120.0), t) is None  # soft, not blurry
    black = scores(sharp_tiles=2.0, luma_mean=4.0, dark_frac=0.97)
    assert reject_reason(black, t) == "dark"  # no detail either, but dark says why
    night = scores(sharp_tiles=2500.0, luma_mean=12.0, dark_frac=0.9)  # sharp lights in the dark
    assert reject_reason(night, t) is None
    white = scores(sharp_tiles=3.0, luma_mean=250.0, bright_frac=0.95)
    assert reject_reason(white, t) == "blown_out"
    screenshot = scores(sharp_tiles=3000.0, luma_mean=235.0, bright_frac=0.9)  # text on white
    assert reject_reason(screenshot, t) is None


def test_the_summary_speaks_of_one_reject_and_one_burst_in_the_singular() -> None:
    def lines(rejects: dict[str, int], bursts: int, hidden: bool = False) -> str:
        culling = CullStats(rejects=rejects, hidden=hidden)
        return "\n".join(_category_lines(CategoryStats(), BurstStats(bursts, 3 * bursts), culling))

    one = lines({"blurry": 1}, 1)
    assert "1 photo flagged as a reject (1 blurry): rejects/ lists it to check, and it stays" in one
    assert "Found 1 burst (3 photos); the best shot goes to the people" in one
    hidden = lines({"blurry": 1}, 0, hidden=True)
    assert "1 photo flagged as a reject (1 blurry): it stays in all/ and goes to rejects/" in hidden
    two = lines({"dark": 1, "blurry": 1}, 2)
    assert "2 photos flagged as rejects (1 dark, 1 blurry): rejects/ lists them" in two
    assert "they stay in their folders" in two
    assert "Found 2 bursts (6 photos); the best shot of each goes" in two


def test_the_export_summary_speaks_of_one_reject_in_the_singular() -> None:
    def summary(rejects: dict[str, int]) -> str:
        result = ExportResult(
            dry_run=False,
            planned=0,
            written=0,
            unchanged=0,
            removed=0,
            linked=0,
            link_fallbacks=0,
            bytes_written=0,
            kept=0,
            unreadable=0,
            near_copies=0,
            duplicates_exported=False,
            left_alone=0,
            problems=[],
            seconds=0.0,
            rejects=rejects,
        )
        return result.summary()

    one = summary({"blurry": 1})
    assert "rejects/ lists 1 file (1 blurry) to check; it stays in people/" in one
    two = summary({"dark": 1, "blurry": 1})
    assert "rejects/ lists 2 files (1 dark, 1 blurry) to check; they stay in people/" in two


@pytest.mark.cull
def test_the_best_shot_of_a_burst() -> None:
    t = load_config().thresholds
    s = {1: scores(sharp_tiles=1000.0), 2: scores(sharp_tiles=970.0), 3: scores(sharp_tiles=400.0)}
    assert best_shot([1, 2, 3], s, {}, Counter(), t) == 1  # 1 and 2 tie on sharpness; 1 is sharper
    assert rank_shots([1, 2, 3], s, {}, Counter(), t) == [1, 2, 3]  # the whole ranking
    assert rank_shots([1, 2, 3], s, {1: "blurry"}, Counter(), t) == [2, 3, 1]
    assert rank_shots([4, 1, 3], s, {}, Counter(), t) == [1, 3, 4]  # no scores: last, by time
    faces = Counter({2: 3, 1: 2})
    assert best_shot([1, 2, 3], s, {}, faces, t) == 2  # tied on sharpness, more faces
    assert best_shot([1, 2, 3], s, {1: "blurry", 2: "dark"}, faces, t) == 3  # rejects last
    s[2] = scores(sharp_tiles=970.0, bright_frac=0.3)
    assert best_shot([1, 2, 3], s, {}, faces, t) == 1  # less clipping beats more faces
    assert best_shot([4, 1], s, {}, Counter(), t) == 1  # a shot without scores ranks last
    assert best_shot([4, 5], s, {}, Counter(), t) == 4  # none has scores: the first shot


@pytest.mark.cull
def test_rejects_and_bursts_get_review_folders(
    tmp_path: Path, color_clip: ColorImageEncoder, word_text: WordTextEncoder
) -> None:
    def photo(image: Image.Image, when: str) -> bytes:
        return jpeg(image, exif=camera_exif(f"2026:07:14 {when}"))

    trip = make_zip(
        tmp_path / "trip.zip",
        [
            (
                "burst1.jpg",
                photo(textured("scenery", 2).filter(ImageFilter.GaussianBlur(1)), "09:00:00"),
            ),
            ("burst2.jpg", photo(textured("scenery", 3), "09:00:01")),  # the sharpest
            (
                "burst3.jpg",
                photo(textured("scenery", 4).filter(ImageFilter.GaussianBlur(8)), "09:00:02"),
            ),
            ("meal.jpg", photo(textured("food", 5), "12:00:00")),
            ("pocket.jpg", photo(flat(2), "13:00:00")),
            ("glare.jpg", photo(flat(253), "14:00:00")),
        ],
    )
    out = tmp_path / "sorted"
    out.mkdir()
    (out / "tripsort.toml").write_text(FAKE_CATEGORIES, encoding="utf-8")
    with Project.open(out) as project:
        project.ingest([trip])
        analyzed = project.analyze()
        assert analyzed.culling.rejects == {"dark": 1, "blown_out": 1, "blurry": 1}
        assert (analyzed.bursts.bursts, analyzed.culling.bursts) == (1, 1)
        summary = analyzed.summary()
        assert "3 photos flagged as rejects (1 dark, 1 blown out, 1 blurry): rejects/" in summary
        assert "they stay in their folders" in summary
        result = project.export()
        assert len(list((out / "all").iterdir())) == 6  # rejects and burst shots stay
        assert names(out / "rejects" / "blurry") == {"burst3.jpg"}
        assert names(out / "rejects" / "dark") == {"pocket.jpg"}
        assert names(out / "rejects" / "blown_out") == {"glare.jpg"}
        (burst,) = (out / "bursts").iterdir()
        assert burst.name.endswith("_burst1")  # named after its first shot
        assert names(burst) == {"burst1.jpg", "burst2.jpg", "burst3.jpg"}
        assert [p.name for p in burst.iterdir() if p.name.startswith("best_")] == [
            "best_" + next(p.name for p in (out / "all").iterdir() if p.name.endswith("burst2.jpg"))
        ]
        # by default rejects stay in the curated folders too; bursts give them the best shot only
        assert names(out / "categories" / "scenery") == {"burst2.jpg"}
        assert names(out / "categories" / "food") == {"meal.jpg"}
        assert names(out / "categories" / "documents") == {"glare.jpg"}  # flagged, still here
        assert result.rejects == {"blown_out": 1, "blurry": 1, "dark": 1}
        assert result.bursts == 1
        assert "rejects/ lists 3 files (1 blown out, 1 blurry, 1 dark) to check" in result.summary()

    with (out / "tripsort.toml").open("a", encoding="utf-8") as fp:
        fp.write("[thresholds]\nhide_rejects = true\n")
    with Project.open(out) as project:
        summary = project.analyze().summary()
        assert "they stay in all/ and go to rejects/, not the people" in summary
        result = project.export()
        assert names(out / "rejects" / "blown_out") == {"glare.jpg"}
        assert names(out / "categories" / "documents") == set()  # hidden: the glare left
        assert names(out / "categories" / "scenery") == {"burst2.jpg"}
        assert "rejects/ holds 3 files (1 blown out, 1 blurry, 1 dark), left out of" in (
            result.summary()
        )
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        by_member = {f["occurrences"][0]["member"]: f for f in manifest["files"]}
        assert by_member["burst3.jpg"]["reject"] == "blurry"
        assert by_member["burst2.jpg"]["burst_best"] is True
        assert by_member["burst1.jpg"]["burst_best"] is False
        assert by_member["meal.jpg"]["burst_best"] is None
        assert by_member["meal.jpg"]["quality"]["sharp_tiles"] > 15


def test_each_person_gets_their_own_best_shot_of_a_burst() -> None:
    people = {1: ["alex"], 2: ["alex", "sam"], 3: ["kim", "sam"], 4: ["kim"]}
    burst = _Burst(best=1, shots=[1, 2, 3, 4], ranked=[1, 3, 2, 4])
    everything = {1, 2, 3, 4}
    # alex is in the best shot; sam and kim are not, and 3 ranks first of their shots
    assert _own_best_shots([burst], people, {}, everything) == {3: ["kim", "sam"]}
    # a reject is never anyone's own best shot
    assert _own_best_shots([burst], people, {3: "blurry"}, everything) == {2: ["sam"], 4: ["kim"]}
    # every shot a reject: nobody gets one
    assert _own_best_shots([burst], people, dict.fromkeys(everything, "dark"), everything) == {}
    # before analyze ranks a burst: the first shot that shows each person, in time order
    unranked = _Burst(best=None, shots=[1, 2], ranked=[1, 2])
    assert _own_best_shots([unranked], people, {}, {1, 2}) == {1: ["alex"], 2: ["sam"]}


def test_someone_the_best_shot_leaves_out_keeps_the_moment(
    tmp_path: Path, color_clip: ColorImageEncoder, marker_faces: MarkerFaceAnalyzer
) -> None:
    def shot(person: int, x: float, when: str, seed: int, blur: float = 0.0) -> bytes:
        image = face_photo([(person, x, 0.2, 0.25)], seed=seed)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        return jpeg(image, exif=camera_exif(f"2026:07:14 {when}"))

    entries = []
    for i in range(4):  # enough photos of persons 1 and 2 for both to be people
        entries.append((f"alex{i}.jpg", shot(1, 0.3, f"1{i}:00:00", seed=10 + i)))
        entries.append((f"sam{i}.jpg", shot(2, 0.3, f"1{i}:30:00", seed=20 + i)))
    # a burst: sam in the sharpest shot, alex only in the two softer ones (sharpness is scored
    # on a gray copy, where sam's yellow square stands out more than alex's orange one, so
    # sam's shot is the sharp one; the other way round, blur would not be enough)
    entries += [
        ("burst1.jpg", shot(2, 0.3, "09:00:00", seed=5)),
        ("burst2.jpg", shot(1, 0.5, "09:00:01", seed=5, blur=2)),
        ("burst3.jpg", shot(1, 0.5, "09:00:02", seed=5, blur=4)),
    ]
    trip = make_zip(tmp_path / "trip.zip", entries)
    with Project.open(tmp_path / "sorted") as project:
        project.ingest([trip])
        assert project.analyze().bursts.bursts == 1
        result = project.export()
        out = project.folder
    folders = [names(p) for p in (out / "people").iterdir()]
    alex = next(f for f in folders if "alex0.jpg" in f)
    sam = next(f for f in folders if "sam0.jpg" in f)
    assert sam == {"sam0.jpg", "sam1.jpg", "sam2.jpg", "sam3.jpg", "burst1.jpg"}  # the best shot
    assert alex == {"alex0.jpg", "alex1.jpg", "alex2.jpg", "alex3.jpg", "burst2.jpg"}  # alex's
    assert {"burst1.jpg"} == names(out / "shots" / "solo") & {"burst1.jpg", "burst2.jpg"}
    (burst,) = (out / "bursts").iterdir()
    assert names(burst) == {"burst1.jpg", "burst2.jpg", "burst3.jpg"}
    assert len(list((out / "all").iterdir())) == 11
    assert result.burst_people == 1
    assert "people/ also holds 1 other shot of bursts" in result.summary()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    ranks = {f["occurrences"][0]["member"]: f["burst_rank"] for f in manifest["files"]}
    assert (ranks["burst1.jpg"], ranks["burst2.jpg"], ranks["burst3.jpg"]) == (1, 2, 3)
    assert ranks["alex0.jpg"] is None
