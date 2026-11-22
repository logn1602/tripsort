"""The model registry and downloads, with local file:// URLs: no test touches the network."""

import dataclasses
import hashlib
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tripsort.cli import app
from tripsort.errors import ModelError
from tripsort.models import download, registry
from tripsort.models.registry import ModelFile


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def start(self, stage: str, total: int, unit: str) -> None:
        self.events.append(("start", (stage, total, unit)))

    def advance(self, amount: int) -> None:
        self.events.append(("advance", amount))

    def finish(self) -> None:
        self.events.append(("finish", None))


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "cache"
    monkeypatch.setenv(download.MODEL_DIR_ENV, str(folder))
    return folder


def local_model(tmp_path: Path, data: bytes = b"weights " * 1000) -> ModelFile:
    source = tmp_path / "upstream" / "fake.onnx"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    return ModelFile(
        name="fake",
        version="1",
        filename="fake-1.onnx",
        url=source.as_uri(),
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        license="MIT",
        purpose="testing",
    )


def test_the_registry_pins_every_model() -> None:
    pinned = (
        f"https://github.com/opencv/opencv_zoo/raw/{registry.OPENCV_ZOO_COMMIT}/",  # a commit
        f"https://github.com/logn1602/tripsort/releases/download/{registry.CLIP_RELEASE}/",
        f"https://github.com/logn1602/tripsort/releases/download/{registry.CLIP_INT8_RELEASE}/",
    )
    every = {*registry.ALL_MODELS, registry.CLIP_IMAGE_FP32, registry.CLIP_IMAGE_INT8}
    for model in every:
        assert model.url.startswith(pinned), model.name
        assert re.fullmatch(r"[0-9a-f]{64}", model.sha256)
        assert model.size > 100_000
        assert model.url.endswith(model.filename)
    assert len({m.filename for m in every}) == len(every)
    assert {m.license for m in registry.FACE_MODELS} == {"MIT", "Apache-2.0"}
    assert {m.license for m in every - set(registry.FACE_MODELS)} == {"MIT"}
    assert registry.CLIP_IMAGE in {registry.CLIP_IMAGE_FP32, registry.CLIP_IMAGE_INT8}


def test_model_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(download.MODEL_DIR_ENV, str(tmp_path / "mine"))
    assert download.model_dir() == tmp_path / "mine"
    monkeypatch.delenv(download.MODEL_DIR_ENV)
    assert "tripsort" in download.model_dir().parts


def test_download_verifies_then_works_offline(cache: Path, tmp_path: Path) -> None:
    model = local_model(tmp_path)
    assert download.model_status(model) == "missing"
    progress = Recorder()
    paths = download.ensure_models([model], progress)
    assert paths == {"fake": cache / "fake-1.onnx"}
    assert paths["fake"].read_bytes() == b"weights " * 1000
    assert progress.events[0] == ("start", ("models", model.size, "bytes"))
    assert sum(a for kind, a in progress.events if kind == "advance") == model.size  # type: ignore[misc]
    assert download.model_status(model) == "ok"
    offline = dataclasses.replace(model, url=(tmp_path / "gone.onnx").as_uri())
    assert download.ensure_models([offline]) == paths  # cached: nothing is downloaded
    assert not list(cache.glob("*.part"))


def test_a_damaged_file_is_downloaded_again(cache: Path, tmp_path: Path) -> None:
    model = local_model(tmp_path)
    cache.mkdir()
    (cache / model.filename).write_bytes(b"truncated")
    assert download.model_status(model) == "corrupt"
    download.ensure_models([model])
    assert download.model_status(model) == "ok"


def test_a_download_that_does_not_match_is_refused(cache: Path, tmp_path: Path) -> None:
    model = dataclasses.replace(local_model(tmp_path), sha256="0" * 64)
    with pytest.raises(ModelError, match="does not match its pinned SHA-256"):
        download.ensure_models([model])
    assert not cache.exists() or not list(cache.iterdir())  # nothing half-written is left
    longer = dataclasses.replace(local_model(tmp_path), size=10)
    with pytest.raises(ModelError, match="expected 10"):
        download.ensure_models([longer])


def test_an_unreachable_model_says_what_to_do(cache: Path, tmp_path: Path) -> None:
    model = dataclasses.replace(local_model(tmp_path), url=(tmp_path / "gone.onnx").as_uri())
    with pytest.raises(ModelError, match="tripsort models download") as caught:
        download.ensure_models([model])
    assert str(cache) in str(caught.value)
    assert not list(cache.glob("*.part"))


def test_models_commands(cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = local_model(tmp_path)
    monkeypatch.setattr(registry, "ALL_MODELS", (model,))
    runner = CliRunner()
    listed = runner.invoke(app, ["models", "list"])
    assert listed.exit_code == 0
    assert "missing" in listed.output
    assert str(cache) in listed.output
    fetched = runner.invoke(app, ["models", "download"])
    assert fetched.exit_code == 0
    assert str(cache / model.filename) in fetched.output
    assert " ok " in runner.invoke(app, ["models", "list"]).output
    monkeypatch.setattr(
        registry, "ALL_MODELS", (dataclasses.replace(model, filename="x.onnx", url="file:///gone"),)
    )
    failed = runner.invoke(app, ["models", "download"])
    assert failed.exit_code == 1
    assert "could not download" in failed.output
