"""The real face models (YuNet and SFace) on public-domain portrait paintings.

Skipped unless the models are cached (`uv run tripsort models download`); CI caches them.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from factories import jpeg, make_zip
from PIL import Image, ImageEnhance, ImageOps
from typer.testing import CliRunner

from tripsort.cli import app
from tripsort.faces.analyzer import load_face_analyzer
from tripsort.faces.types import FaceAnalyzer

pytestmark = pytest.mark.models

PAINTINGS = Path(__file__).parent / "fixtures" / "paintings"
NAMES = ("mona_lisa", "girl_with_a_pearl_earring", "george_washington")


def painting(name: str) -> Image.Image:
    with Image.open(PAINTINGS / f"{name}.jpg") as image:
        return image.convert("RGB")


def augmented(image: Image.Image) -> dict[str, Image.Image]:
    w, h = image.size
    return {
        "flip": ImageOps.mirror(image),
        "crop": image.crop((int(w * 0.08), int(h * 0.05), int(w * 0.92), int(h * 0.85))),
        "darker": ImageEnhance.Brightness(image).enhance(0.6),
        "brighter": ImageEnhance.Brightness(image).enhance(1.4),
        "smaller": image.resize((w * 400 // h, 400), Image.Resampling.LANCZOS),
    }


@pytest.fixture(scope="module")
def analyzer() -> FaceAnalyzer:
    return load_face_analyzer()


def test_each_painting_has_one_clear_face(analyzer: FaceAnalyzer) -> None:
    for name in NAMES:
        faces = analyzer.analyze(np.asarray(painting(name)))
        assert sum(f.score >= 0.8 for f in faces) == 1, name
        face = faces[0]
        assert face.score >= 0.85, name
        assert 0.15 <= face.h <= 0.5, name
        for x, y in face.landmarks:
            assert face.x <= x <= face.x + face.w, name
            assert face.y <= y <= face.y + face.h, name
        assert face.embedding.shape == (128,)
        assert face.embedding.dtype == np.float32
        assert float(np.linalg.norm(face.embedding)) == pytest.approx(1.0, abs=1e-5)


def test_copies_of_a_painting_match_and_other_paintings_do_not(analyzer: FaceAnalyzer) -> None:
    embeddings: dict[str, list[np.ndarray]] = {}
    same = []
    for name in NAMES:
        original = analyzer.analyze(np.asarray(painting(name)))[0].embedding
        variants = [
            analyzer.analyze(np.asarray(img))[0].embedding
            for img in augmented(painting(name)).values()
        ]
        same += [float(original @ v) for v in variants]
        embeddings[name] = [original, *variants]
    different = [
        float(a @ b)
        for i, x in enumerate(NAMES)
        for y in NAMES[i + 1 :]
        for a in embeddings[x]
        for b in embeddings[y]
    ]
    # Measured on Windows: same >= 0.89, different <= 0.40.
    assert min(same) >= 0.75
    assert max(different) <= 0.55
    assert min(same) - max(different) >= 0.3


def test_run_sorts_paintings_into_people_and_refs_name_them(tmp_path: Path) -> None:
    """Five photos of each painting (crops change the aspect ratio, so none is a near-copy of
    another), a refs folder with a brightened copy of each, then `tripsort run`."""
    entries = []
    refs = tmp_path / "refs"
    refs.mkdir()
    short = {
        "mona_lisa": "mona",
        "girl_with_a_pearl_earring": "pearl",
        "george_washington": "george",
    }
    for name in NAMES:
        image = painting(name)
        w, h = image.size
        top = image.crop((0, 0, w, int(h * 0.75)))
        photos = {
            "orig": image,
            "flip": ImageOps.mirror(image),
            "top": top,
            "middle": image.crop((int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.7))),
            "top-flip": ImageOps.mirror(top),
        }
        entries += [(f"{name}-{kind}.jpg", jpeg(img)) for kind, img in photos.items()]
        ImageEnhance.Brightness(image).enhance(1.3).save(refs / f"{short[name]}.jpg")
    zip_path = make_zip(tmp_path / "paintings.zip", entries)
    out = tmp_path / "sorted"

    result = CliRunner().invoke(app, ["run", str(zip_path), "-o", str(out), "--refs", str(refs)])
    assert result.exit_code == 0, result.output

    folders = {p.name: sorted(f.name for f in p.iterdir()) for p in (out / "people").iterdir()}
    assert sorted(folders) == ["george", "mona", "pearl"]
    for name in NAMES:
        files = folders[short[name]]
        assert len(files) >= 3
        assert all(name in f for f in files), files
    assert len(list((out / "shots" / "solo").iterdir())) == sum(map(len, folders.values()))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert {p["name"]: p["named_by"] for p in manifest["people"]} == {
        "george": "refs",
        "mona": "refs",
        "pearl": "refs",
    }
