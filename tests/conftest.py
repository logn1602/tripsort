import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from factories import camera_exif, heic, jpeg, make_zip, photo, whatsapp_copy
from fake_clip import ColorImageEncoder, NullImageEncoder, WordTextEncoder
from fake_faces import MarkerFaceAnalyzer, NullFaceAnalyzer

import tripsort.clip.encoder
import tripsort.commands
import tripsort.config
import tripsort.faces.analyzer
from tripsort.api import Project
from tripsort.models.download import model_status
from tripsort.models.registry import FACE_MODELS


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked `models` run only when the real face models are cached."""
    if not any("models" in item.keywords for item in items):
        return
    if all(model_status(m) == "ok" for m in FACE_MODELS):
        return
    skip = pytest.mark.skip(reason="face models not downloaded: `uv run tripsort models download`")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Show the lines tests leave with ``record_property("summary", ...)``, such as how close
    the int8 image encoder came to fp32 on this CPU."""
    for outcome in ("passed", "failed"):
        for report in terminalreporter.stats.get(outcome, []):
            for name, value in getattr(report, "user_properties", ()):
                if name == "summary":
                    terminalreporter.write_line(str(value))


@pytest.fixture(autouse=True)
def _no_rejects(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated test photos are flat colors, which the reject rules call blurry, so
    rejects are off unless a test is marked `cull`."""
    if request.node.get_closest_marker("cull"):
        return
    off = {"reject_max_sharpness": "0.0", "reject_max_luma": "0.0", "reject_min_bright_frac": "1.0"}
    lines = []
    for line in tripsort.config.default_config_text().splitlines():
        key = line.split("=")[0].strip()
        lines.append(f"{key} = {off[key]}" if key in off else line)
    text = "\n".join(lines) + "\n"
    monkeypatch.setattr(tripsort.config, "default_config_text", lambda: text)


@pytest.fixture(autouse=True)
def _no_real_models(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Outside `models` tests, the models are never loaded and nothing is downloaded: the
    model folder is empty, the face analyzer finds no faces, and the CLIP encoders are the
    fakes in fake_clip.py."""
    if request.node.get_closest_marker("models"):
        return
    monkeypatch.setenv("TRIPSORT_MODEL_DIR", str(tmp_path_factory.mktemp("no-models")))
    monkeypatch.setattr(
        tripsort.faces.analyzer, "load_face_analyzer", lambda progress=None: NullFaceAnalyzer()
    )
    monkeypatch.setattr(
        tripsort.clip.encoder, "load_image_encoder", lambda progress=None: NullImageEncoder()
    )
    monkeypatch.setattr(
        tripsort.clip.encoder, "load_text_encoder", lambda progress=None: WordTextEncoder()
    )


@pytest.fixture
def marker_faces(monkeypatch: pytest.MonkeyPatch) -> MarkerFaceAnalyzer:
    """Faces from colored squares (see fake_faces.face_photo) instead of the real models."""
    analyzer = MarkerFaceAnalyzer()
    monkeypatch.setattr(
        tripsort.faces.analyzer, "load_face_analyzer", lambda progress=None: analyzer
    )
    return analyzer


@pytest.fixture
def color_clip(monkeypatch: pytest.MonkeyPatch) -> ColorImageEncoder:
    """CLIP embeddings from a photo's average color (see fake_clip.py) instead of the model."""
    encoder = ColorImageEncoder()
    monkeypatch.setattr(tripsort.clip.encoder, "load_image_encoder", lambda progress=None: encoder)
    return encoder


@pytest.fixture
def word_text(monkeypatch: pytest.MonkeyPatch) -> WordTextEncoder:
    """The fake text encoder (see fake_clip.py), kept so a test can see what it embedded."""
    encoder = WordTextEncoder()
    monkeypatch.setattr(tripsort.clip.encoder, "load_text_encoder", lambda progress=None: encoder)
    return encoder


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Project]:
    with Project.open(tmp_path / "sorted") as proj:
        yield proj


@pytest.fixture
def uvx(monkeypatch: pytest.MonkeyPatch) -> str:
    """Messages as after a uvx run without the video extra (test_commands.py checks how a
    uvx run is recognized). Returns the command prefix they should print."""
    monkeypatch.setattr(tripsort.commands, "run_by_uvx", lambda: True)
    monkeypatch.setattr(tripsort.commands, "has_video_extra", lambda: False)
    return f"uvx --python {sys.version_info.major}.{sys.version_info.minor} tripsort"


@pytest.fixture
def trip(tmp_path: Path) -> dict[str, Path]:
    """Two zips from a small trip: exact duplicates across zips, a WhatsApp copy, a HEIC,
    junk, an unsupported file, a video, and a truncated JPEG."""
    a = jpeg(photo(1), exif=camera_exif("2026:07:14 10:23:45", subsec="120"))
    b = jpeg(photo(2), exif=camera_exif("2026:07:14 11:00:00", orientation=6))
    c = heic(
        photo(3, (480, 360)),
        exif=camera_exif("2026:07:15 09:00:00", make="Apple", model="iPhone 15"),
    )
    mine = make_zip(
        tmp_path / "in" / "mine.zip",
        [
            ("DCIM/PXL_20260714_082345120.jpg", a),
            ("DCIM/PXL_20260714_090000000.jpg", b),
            ("DCIM/IMG_0001.HEIC", c),
            ("__MACOSX/DCIM/._PXL_20260714_082345120.jpg", b"junk"),
            (".DS_Store", b"junk"),
            ("notes.txt", b"hello"),
            ("DCIM/VID_20260714_101010.mp4", b"\x00\x00\x00\x18ftypmp42" + bytes(4000)),
            ("DCIM/broken.jpg", a[:3000]),
        ],
    )
    sam = make_zip(
        tmp_path / "in" / "sam.zip",
        [
            ("WhatsApp Images/IMG-20260714-WA0001.jpg", whatsapp_copy(a)),
            ("Camera/same_as_mine.jpg", b),
            (
                "Camera/IMG_20260716_120000.jpg",
                jpeg(photo(5), exif=camera_exif("2026:07:16 12:00:00", model="SM-S911B")),
            ),
        ],
    )
    return {"mine": mine, "sam": sam}
