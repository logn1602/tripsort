"""CLIP: the tokenizer, the preprocessing, and embedding photos at ingest.

Tests marked `models` use the real (cached) model files and compare them with what open_clip
gives (tests/fixtures/clip/, written by tools/export_clip.py).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from factories import jpeg, make_zip, photo
from fake_clip import ColorImageEncoder
from PIL import Image
from typer.testing import CliRunner

import tripsort.clip.encoder
from tripsort.api import Project
from tripsort.cli import app
from tripsort.clip import tokenizer
from tripsort.clip.preprocess import MEAN, STD, clip_input
from tripsort.clip.types import BATCH_SIZE, CLIP_MODEL_ID, IMAGE_SIZE, unit_rows
from tripsort.errors import ModelError
from tripsort.ingest.decode import decode_oriented

FIXTURES = Path(__file__).parent / "fixtures"


def solid(rgb: tuple[int, int, int], size: tuple[int, int] = (640, 480)) -> bytes:
    return jpeg(Image.new("RGB", size, rgb))


# tokenizer rules (no vocabulary needed)


def test_text_is_cleaned_like_open_clip() -> None:
    assert tokenizer.clean("  A  Photo\tof\na DOG  ") == "a photo of a dog"
    assert tokenizer.clean("rock &amp;amp; roll") == "rock & roll"  # unescaped twice
    assert tokenizer.clean(f"it{chr(0x2019)}s {chr(0xFB01)}ne") == "it's fine"  # quote, ligature
    assert tokenizer.clean(f"{chr(0xFF21)}{chr(0xFF22)}{chr(0x3000)}c") == "ab c"  # full width
    assert tokenizer.clean("cafe\u0301") == "caf\u00e9"  # NFC


def test_text_is_split_by_unicode_category() -> None:
    assert tokenizer.split_words("it's a dog's 12a!'s") == [
        "it",
        "'s",
        "a",
        "dog",
        "'s",
        "1",
        "2",
        "a",
        "!'",
        "s",
    ]
    assert tokenizer.split_words("<start_of_text>hi") == ["<start_of_text>", "hi"]
    assert tokenizer.split_words("z\u00fcrich \u00bd") == ["z\u00fcrich", "\u00bd"]


def test_every_byte_has_a_printable_character() -> None:
    table = tokenizer.bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256
    assert all(not c.isspace() for c in table.values())


# preprocessing


def test_preprocessing_matches_clip_shapes_and_normalization() -> None:
    gray = np.full((300, 500, 3), 128, dtype=np.uint8)
    out = clip_input(gray)
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == np.float32
    expected = (128 / 255 - MEAN) / STD
    assert np.allclose(out.mean(axis=(1, 2)), expected, atol=1e-3)
    portrait = np.zeros((2000, 1000, 3), dtype=np.uint8)
    portrait[:400] = 255  # a white top fifth, cropped away: only the center is kept
    assert np.allclose(clip_input(portrait), (0 - MEAN[:, None, None]) / STD[:, None, None])
    assert clip_input(np.zeros((3, 900, 3), dtype=np.uint8)).shape == (3, 224, 224)


def test_unit_rows() -> None:
    rows = unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(rows, [[0.6, 0.8], [0.0, 0.0]])


# embedding at ingest


def embeddings(project: Project) -> dict[int, np.ndarray]:
    return {
        int(media_id): np.frombuffer(blob, dtype=np.float32)
        for media_id, blob in project.connection.execute(
            "SELECT media_id, embedding FROM clip_embeddings WHERE model_id = ?",
            (CLIP_MODEL_ID,),
        )
    }


def test_photos_are_embedded_in_the_same_decode(
    project: Project, color_clip: ColorImageEncoder, tmp_path: Path
) -> None:
    entries = [(f"p{i}.jpg", jpeg(photo(i))) for i in range(BATCH_SIZE + 3)]
    entries.append(("v.mp4", b"\x00\x00\x00\x18ftypmp42" + bytes(2000)))
    result = project.ingest([make_zip(tmp_path / "trip.zip", entries)])
    assert (result.clip_embedded, result.clip_caught_up, result.clip_pending) == (19, 0, 0)
    assert color_clip.calls == 2  # one full batch of 16, then the 3 left at the commit
    found = embeddings(project)
    assert len(found) == 19
    assert all(abs(float(np.linalg.norm(v)) - 1) < 1e-5 for v in found.values())
    statuses = project.connection.execute(
        "SELECT status, COUNT(*) FROM clip_scans GROUP BY status"
    ).fetchall()
    assert [tuple(row) for row in statuses] == [("ok", 19)]  # the video is not an image
    again = project.ingest([tmp_path / "trip.zip"])
    assert (again.clip_embedded, color_clip.calls) == (0, 2)  # nothing embedded twice


def test_photos_are_embedded_later_when_the_model_was_missing(
    project: Project, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("could not download clip-vit-b-32-openai-image-fp32.onnx")

    monkeypatch.setattr(tripsort.clip.encoder, "load_image_encoder", offline)
    trip = make_zip(
        tmp_path / "trip.zip", [(f"p{i}.jpg", solid((10 * i, 90, 160))) for i in range(5)]
    )
    first = project.ingest([trip])
    assert first.clip_problem is not None
    assert first.clip_pending == 5
    assert "Photos were not embedded for categories and bursts" in first.summary()
    assert "5 photos still wait for a CLIP embedding" in first.summary()

    encoder = ColorImageEncoder()
    monkeypatch.setattr(tripsort.clip.encoder, "load_image_encoder", lambda progress=None: encoder)
    caught_up = project.ingest()
    assert (caught_up.clip_embedded, caught_up.clip_caught_up, caught_up.clip_pending) == (5, 5, 0)
    assert "Embedded 5 photos from earlier zips for categories and bursts." in caught_up.summary()
    assert project.ingest().clip_embedded == 0  # nothing left


def test_an_offline_clip_model_makes_the_run_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("offline")

    monkeypatch.setattr(tripsort.clip.encoder, "load_image_encoder", offline)
    trip = make_zip(tmp_path / "trip.zip", [("p.jpg", jpeg(photo(1)))])
    result = CliRunner().invoke(app, ["run", str(trip), "-o", str(tmp_path / "sorted")])
    assert result.exit_code == 2
    assert (tmp_path / "sorted" / "all").is_dir()  # dedupe and export still work


def test_an_encoder_failure_is_recorded_not_fatal(
    project: Project, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Broken(ColorImageEncoder):
        def embed(self, batch: np.ndarray) -> np.ndarray:
            raise RuntimeError("the model broke")

    monkeypatch.setattr(tripsort.clip.encoder, "load_image_encoder", lambda progress=None: Broken())
    result = project.ingest([make_zip(tmp_path / "t.zip", [("p.jpg", jpeg(photo(1)))])])
    assert result.clip_embedded == 0
    assert result.clip_pending == 0  # scanned, and recorded as failed: not tried every run
    (status, detail) = project.connection.execute(
        "SELECT status, detail FROM clip_scans"
    ).fetchone()
    assert (status, detail) == ("failed", "RuntimeError: the model broke")


# the real models


@pytest.mark.models
def test_the_tokenizer_matches_open_clip() -> None:
    from tripsort.models.download import ensure_models
    from tripsort.models.registry import CLIP_VOCAB

    vocab = ensure_models([CLIP_VOCAB])[CLIP_VOCAB.name]
    tok = tokenizer.Tokenizer(vocab)
    cases = json.loads((FIXTURES / "clip" / "tokens.json").read_text(encoding="utf-8"))["cases"]
    for case in cases:
        got = [int(t) for t in tok([case["text"]])[0] if t != 0]
        assert got == case["input_ids"], ascii(case["text"])


@pytest.mark.models
def test_the_image_encoders_match_open_clip_on_the_paintings() -> None:
    from tripsort.clip.encoder import load_image_encoder
    from tripsort.models.registry import CLIP_IMAGE_FP32, CLIP_IMAGE_INT8

    refs = json.loads((FIXTURES / "clip" / "paintings.json").read_text(encoding="utf-8"))
    batch = np.stack(
        [
            clip_input(decode_oriented((FIXTURES / "paintings" / r["file"]).read_bytes()))
            for r in refs["images"]
        ]
    )
    want = unit_rows(np.array([r["embedding"] for r in refs["images"]]))
    for model, floor in ((CLIP_IMAGE_FP32, 0.999), (CLIP_IMAGE_INT8, 0.98)):
        got = load_image_encoder(model=model).embed(batch)
        assert float((got * want).sum(axis=1).min()) >= floor, model.name


@pytest.mark.models
def test_the_text_encoder_puts_prompts_near_matching_photos() -> None:
    from tripsort.clip.encoder import load_image_encoder, load_text_encoder

    text = load_text_encoder().embed(["a painting of a woman", "a photo of a dog", "a city map"])
    assert np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-5)
    mona = clip_input(decode_oriented((FIXTURES / "paintings" / "mona_lisa.jpg").read_bytes()))
    image = load_image_encoder().embed(mona[None])
    assert int(np.argmax(image @ text.T)) == 0
