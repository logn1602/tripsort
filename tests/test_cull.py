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
from PIL import Image, ImageFilter

from tripsort.api import Project
from tripsort.config import load_config
from tripsort.cull.analysis import _Scores, best_shot, reject_reason
from tripsort.cull.quality import measure

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
    assert reject_reason(scores(sharp_tiles=40.0), t) is None  # soft, not blurry
    black = scores(sharp_tiles=2.0, luma_mean=4.0, dark_frac=0.97)
    assert reject_reason(black, t) == "dark"  # no detail either, but dark says why
    night = scores(sharp_tiles=400.0, luma_mean=12.0, dark_frac=0.9)  # lights in the dark
    assert reject_reason(night, t) is None
    white = scores(sharp_tiles=3.0, luma_mean=250.0, bright_frac=0.95)
    assert reject_reason(white, t) == "blown_out"
    screenshot = scores(sharp_tiles=900.0, luma_mean=235.0, bright_frac=0.9)  # text on white
    assert reject_reason(screenshot, t) is None


@pytest.mark.cull
def test_the_best_shot_of_a_burst() -> None:
    t = load_config().thresholds
    s = {1: scores(sharp_tiles=1000.0), 2: scores(sharp_tiles=950.0), 3: scores(sharp_tiles=400.0)}
    assert best_shot([1, 2, 3], s, {}, Counter(), t) == 1  # 1 and 2 tie on sharpness; 1 is sharper
    faces = Counter({2: 3, 1: 2})
    assert best_shot([1, 2, 3], s, {}, faces, t) == 2  # tied on sharpness, more faces
    assert best_shot([1, 2, 3], s, {1: "blurry", 2: "dark"}, faces, t) == 3  # rejects last
    s[2] = scores(sharp_tiles=950.0, bright_frac=0.3)
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
    (tmp_path / "sorted").mkdir()
    (tmp_path / "sorted" / "tripsort.toml").write_text(FAKE_CATEGORIES, encoding="utf-8")
    with Project.open(tmp_path / "sorted") as project:
        project.ingest([trip])
        analyzed = project.analyze()
        assert analyzed.culling.rejects == {"dark": 1, "blown_out": 1, "blurry": 1}
        assert (analyzed.bursts.bursts, analyzed.culling.bursts) == (1, 1)
        summary = analyzed.summary()
        assert "3 photos flagged as rejects (1 dark, 1 blown out, 1 blurry)" in summary
        result = project.export()
        out = project.folder
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
        # the curated folders hold the best shot only, and no reject
        assert names(out / "categories" / "scenery") == {"burst2.jpg"}
        assert names(out / "categories" / "food") == {"meal.jpg"}
        assert not (out / "categories" / "documents").exists()  # the glare is a reject
        assert result.rejects == {"blown_out": 1, "blurry": 1, "dark": 1}
        assert result.bursts == 1
        assert "rejects/ holds 3 files (1 blown out, 1 blurry, 1 dark)" in result.summary()
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        by_member = {f["occurrences"][0]["member"]: f for f in manifest["files"]}
        assert by_member["burst3.jpg"]["reject"] == "blurry"
        assert by_member["burst2.jpg"]["burst_best"] is True
        assert by_member["burst1.jpg"]["burst_best"] is False
        assert by_member["meal.jpg"]["burst_best"] is None
        assert by_member["meal.jpg"]["quality"]["sharp_tiles"] > 15
