"""Categories, the selfie hint, and bursts from CLIP embeddings.

The fakes in fake_clip.py stand in for CLIP: a photo painted in one of HUES gets that
kind's embedding, and a prompt that names a kind gets the same direction.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from factories import camera_exif, jpeg, make_zip
from fake_clip import HUES, ColorImageEncoder, WordTextEncoder, axis
from fake_faces import MarkerFaceAnalyzer, marker_color
from PIL import Image, ImageDraw

import tripsort.clip.encoder
from tripsort.api import Project
from tripsort.clip.bursts import Shot, group_bursts
from tripsort.clip.categorize import primaries, shares
from tripsort.clip.prompts import config_prompts, shipped
from tripsort.clip.types import TEXT_MODEL_ID, unit_rows
from tripsort.config import load_config
from tripsort.errors import ModelError

FAKE_CATEGORIES = """
[categories.scenery]
prompts = ["scenery"]
[categories.animals]
prompts = ["animals"]
[categories.food]
prompts = ["food", "more food"]
[categories.documents]
prompts = ["documents"]
[categories.other]
prompts = ["anything"]
[people]
selfie_min_score = 0.8
selfie_prompts = ["selfie"]
not_selfie_prompts = ["anything"]
"""


def painted(kind: str, seed: int, size: tuple[int, int] = (800, 600)) -> Image.Image:
    """A photo of one kind: its HUES color, with a small seed-dependent mark."""
    image = Image.new("RGB", size, HUES[kind][0])
    ImageDraw.Draw(image).rectangle(
        (seed * 7 % 700, 10, seed * 7 % 700 + 20, 30), fill=(60, 60, 60)
    )
    return image


def shot(kind: str, seed: int, when: str, model: str = "Pixel 7", **exif: object) -> bytes:
    return jpeg(painted(kind, seed), exif=camera_exif(f"2026:07:14 {when}", model=model, **exif))


def project_with(tmp_path: Path, config: str) -> Project:
    folder = tmp_path / "sorted"
    folder.mkdir()
    (folder / "tripsort.toml").write_text(config, encoding="utf-8")
    return Project.open(folder)


def members(folder: Path) -> set[str]:
    return {p.name.rsplit("_", 1)[-1] for p in folder.iterdir()} if folder.is_dir() else set()


# the default prompts ship with their embeddings


def test_the_shipped_embeddings_match_the_default_prompts() -> None:
    """Fails when a default prompt changes without `uv run tools/embed_default_prompts.py`."""
    data = json.loads(
        (Path(tripsort.clip.__file__).parent / "default_prompts.json").read_text(encoding="utf-8")
    )
    assert data["model_id"] == TEXT_MODEL_ID
    assert sorted(data["prompts"]) == sorted(config_prompts(load_config()))
    assert all(abs(float(np.linalg.norm(v)) - 1) < 1e-4 for v in shipped().values())


@pytest.mark.models
def test_the_shipped_embeddings_are_the_text_encoders() -> None:
    from tripsort.clip.encoder import load_text_encoder

    prompts = config_prompts(load_config())
    fresh = load_text_encoder().embed(prompts)
    for prompt, vector in zip(prompts, fresh, strict=True):
        assert float(vector @ shipped()[prompt]) > 0.9999, prompt


# the math


def test_shares_add_up_to_one_and_follow_the_prompts() -> None:
    images = unit_rows(np.stack([axis(1) + axis(0, 0.2), axis(2), np.zeros(512, np.float32)]))
    sets = [np.stack([axis(1)]), np.stack([axis(2), axis(3)]), np.stack([axis(0)])]
    scores = shares(images, sets)
    assert np.allclose(scores.sum(axis=1), 1)
    assert int(np.argmax(scores[0])) == 0
    assert np.allclose(scores[2], 1 / 3)  # a zero embedding: every category the same
    labels = ["scenery", "food", "other"]
    assert primaries(scores, labels, 0.3) == ["scenery", "food", "other"]
    below = np.array([[0.25, 0.5, 0.25]])  # food on top, under the threshold
    assert primaries(below, labels, 0.6) == ["other"]
    tie = np.array([[0.4, 0.2, 0.4]])  # a tie with other goes to other
    assert primaries(tie, labels, 0.3) == ["other"]
    assert shares(np.zeros((0, 512), np.float32), sets).shape == (0, 3)


# categories end to end


def test_photos_are_sorted_into_category_folders(
    tmp_path: Path, color_clip: ColorImageEncoder, word_text: WordTextEncoder
) -> None:
    trip = make_zip(
        tmp_path / "trip.zip",
        [
            ("view1.jpg", shot("scenery", 1, "09:00:00")),
            ("view2.jpg", shot("scenery", 2, "09:10:00")),
            ("meal.jpg", shot("food", 3, "12:00:00")),
            ("ticket.jpg", shot("documents", 4, "15:00:00")),
            ("dog.jpg", shot("animals", 5, "16:00:00")),
        ],
    )
    with project_with(tmp_path, FAKE_CATEGORIES) as project:
        project.ingest([trip])
        analyzed = project.analyze()
        assert analyzed.categories.photos == {
            "scenery": 2,
            "animals": 1,
            "food": 1,
            "documents": 1,
            "other": 0,
        }
        assert (
            "Sorted 5 photos by category: 2 scenery, 1 animals, 1 food, 1 documents; 0 other."
            in (analyzed.summary())
        )
        assert sorted(set(word_text.texts)) == [
            "animals",
            "anything",
            "documents",
            "food",
            "more food",
            "scenery",
            "selfie",
        ]
        result = project.export()
        out = project.folder
        assert members(out / "categories" / "scenery") == {"view1.jpg", "view2.jpg"}
        assert members(out / "categories" / "food") == {"meal.jpg"}
        assert not (out / "categories" / "other").exists()  # other gets no folder
        assert result.categories == {"animals": 1, "documents": 1, "food": 1, "scenery": 2}
        assert "categories/ holds 1 animals, 1 documents, 1 food, 2 scenery." in result.summary()
        for link in (out / "categories" / "scenery").iterdir():
            assert link.samefile(out / "all" / link.name)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        meal = next(f for f in manifest["files"] if f["occurrences"][0]["member"] == "meal.jpg")
        assert meal["category"] == "food"
        assert set(meal["category_scores"]) == {"scenery", "animals", "food", "documents", "other"}
        assert abs(sum(meal["category_scores"].values()) - 1) < 1e-3
        assert manifest["summary"]["categories"]["scenery"] == 2
        assert manifest["categories"]["food"] == ["food", "more food"]
        word_text.texts.clear()
        project.analyze()
        assert word_text.texts == []  # prompt embeddings are cached in the index

    (tmp_path / "sorted" / "tripsort.toml").write_text(
        FAKE_CATEGORIES.replace('prompts = ["food", "more food"]', "prompts = []"),
        encoding="utf-8",
    )
    with Project.open(tmp_path / "sorted") as again:
        result = again.export()  # food is dropped: its folder goes, its photo goes elsewhere
        assert not (again.folder / "categories" / "food").exists()
        assert "food" not in result.categories


def test_the_default_prompts_need_no_text_encoder(
    project: Project, color_clip: ColorImageEncoder, word_text: WordTextEncoder, tmp_path: Path
) -> None:
    project.ingest([make_zip(tmp_path / "t.zip", [("a.jpg", shot("food", 1, "09:00:00"))])])
    analyzed = project.analyze()
    assert analyzed.categories.scored == 1
    assert word_text.texts == []


def test_without_the_text_encoder_custom_categories_wait(
    tmp_path: Path, color_clip: ColorImageEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("could not download clip-vit-b-32-openai-text-fp32.onnx")

    monkeypatch.setattr(tripsort.clip.encoder, "load_text_encoder", offline)
    trip = make_zip(tmp_path / "t.zip", [("a.jpg", shot("food", 1, "09:00:00"))])
    with project_with(tmp_path, FAKE_CATEGORIES) as project:
        project.ingest([trip])
        analyzed = project.analyze()
        assert analyzed.categories.problem is not None
        assert "Categories were not scored: could not download" in analyzed.summary()
        assert not (project.folder / "categories").exists()
        project.export()
        assert not (project.folder / "categories").exists()


def test_photos_without_an_embedding_are_counted(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("offline")

    monkeypatch.setattr(tripsort.clip.encoder, "load_image_encoder", offline)
    project.ingest([make_zip(tmp_path / "t.zip", [("a.jpg", shot("food", 1, "09:00:00"))])])
    analyzed = project.analyze()
    assert (analyzed.categories.scored, analyzed.categories.not_embedded) == (0, 1)
    assert "1 photo without a CLIP embedding yet" in analyzed.summary()


# the selfie hint


def selfie_photo(seed: int, lens: str | None) -> bytes:
    image = Image.new("RGB", (1200, 900), HUES["selfie"][0])
    ImageDraw.Draw(image).rectangle((420, 180, 690, 450), fill=marker_color(1))
    exif = camera_exif(f"2026:07:14 1{seed}:00:00", **({"lens": lens} if lens else {}))
    return jpeg(image, exif=exif)


def test_the_selfie_hint_counts_only_when_the_lens_does_not_say(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    marker_faces: MarkerFaceAnalyzer,
) -> None:
    trip = make_zip(
        tmp_path / "t.zip",
        [
            ("unknown.jpg", selfie_photo(1, None)),
            ("back.jpg", selfie_photo(2, "Pixel 7 back camera 6.8mm f/1.9")),
            ("front.jpg", selfie_photo(3, "iPhone 15 front camera 2.69mm f/1.9")),
        ],
    )
    with project_with(tmp_path, FAKE_CATEGORIES) as project:
        project.ingest([trip])
        project.export()
        assert members(project.folder / "shots" / "selfie") == {"unknown.jpg", "front.jpg"}
        hint = dict(
            project.connection.execute(
                "SELECT s.member_path, m.clip_selfie FROM media m JOIN occurrences s "
                "ON s.media_id = m.id"
            ).fetchall()
        )
        assert hint["back.jpg"] > 0.9  # CLIP thinks so too, but the lens model wins


# bursts


def burst_shot(media_id: int, seconds: float, vector: list[float], camera: str = "a") -> Shot:
    return Shot(media_id, camera, seconds, unit_rows(np.array([vector]))[0])


def test_bursts_group_alike_shots_seconds_apart_on_one_camera() -> None:
    same, near, other = [1.0, 0.0], [0.97, 0.243], [0.0, 1.0]
    shots = [
        burst_shot(1, 0.0, same),
        burst_shot(2, 1.0, near),
        burst_shot(3, 2.5, other),  # different, inside the window: starts its own group
        burst_shot(4, 9.5, same),
        burst_shot(5, 10.5, same),  # 10.5 s after the first shot: outside the window
        burst_shot(6, 11.0, other),
        burst_shot(7, 1.5, same, camera="b"),  # another camera never joins
    ]
    assert group_bursts(shots, 0.92, 10.0) == [[1, 2, 4], [3, 6]]
    assert group_bursts(shots, 0.92, 12.0) == [[1, 2, 4, 5], [3, 6]]
    assert group_bursts(shots, 0.999, 10.0) == [[1, 4], [3, 6]]  # near is 0.97 like same
    assert group_bursts([], 0.92, 10.0) == []


def test_a_slow_pan_does_not_chain_into_one_burst() -> None:
    # each shot 12 degrees on: 0.978 to its neighbor, but 0.914 two shots apart
    angles = np.radians([0, 12, 24, 36, 48])
    shots = [
        burst_shot(i + 1, float(i), [float(np.cos(a)), float(np.sin(a))])
        for i, a in enumerate(angles)
    ]
    assert group_bursts(shots, 0.92, 10.0) == [[1, 2], [3, 4]]  # a chain would take all five


def test_bursts_end_to_end(tmp_path: Path, color_clip: ColorImageEncoder) -> None:
    trip = make_zip(
        tmp_path / "t.zip",
        [
            ("b1.jpg", shot("scenery", 1, "09:00:00")),
            ("b2.jpg", shot("scenery", 2, "09:00:01")),
            ("b3.jpg", shot("scenery", 3, "09:00:02")),
            ("later.jpg", shot("scenery", 4, "09:05:00")),
            ("meal.jpg", shot("food", 5, "09:00:03")),
            ("other_phone.jpg", shot("scenery", 6, "09:00:01", model="iPhone 15")),
        ],
    )
    with Project.open(tmp_path / "sorted") as project:
        project.ingest([trip])
        analyzed = project.analyze()
        assert (analyzed.bursts.bursts, analyzed.bursts.photos) == (1, 3)
        assert "Found 1 burst (3 photos); every shot stays in all/." in analyzed.summary()
        project.export()
        assert len(list((project.folder / "all").iterdir())) == 6
        manifest = json.loads((project.folder / "manifest.json").read_text(encoding="utf-8"))
        burst = {f["occurrences"][0]["member"]: f["burst"] for f in manifest["files"]}
        assert burst == {
            "b1.jpg": 1,
            "b2.jpg": 1,
            "b3.jpg": 1,
            "later.jpg": None,
            "meal.jpg": None,
            "other_phone.jpg": None,
        }
        assert (manifest["summary"]["bursts"], manifest["summary"]["photos_in_bursts"]) == (1, 3)
