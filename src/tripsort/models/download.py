"""Download model files into the user cache, verify them, and find them again offline.

The cache is ``platformdirs.user_cache_dir("tripsort")/models`` unless ``TRIPSORT_MODEL_DIR``
names another folder. A file is used only when its size and SHA-256 match the registry, and
it is checked again every time it is loaded, so a damaged file is downloaded again instead of
being used.
"""

import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import platformdirs

from tripsort.errors import ModelError
from tripsort.models.registry import ModelFile
from tripsort.progress import NullProgress, Progress

MODEL_DIR_ENV = "TRIPSORT_MODEL_DIR"
_CHUNK = 1 << 20
_TIMEOUT_SECONDS = 60


def model_dir() -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(platformdirs.user_cache_dir("tripsort", appauthor=False)) / "models"


def model_path(model: ModelFile) -> Path:
    return model_dir() / model.filename


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def model_status(model: ModelFile) -> str:
    """'ok', 'missing', or 'corrupt' (present, but its size or SHA-256 is wrong)."""
    path = model_path(model)
    if not path.is_file():
        return "missing"
    if path.stat().st_size != model.size or file_sha256(path) != model.sha256:
        return "corrupt"
    return "ok"


def ensure_models(models: Iterable[ModelFile], progress: Progress | None = None) -> dict[str, Path]:
    """The local path of each model (keyed by name), downloading the ones not cached yet.

    Raises ``ModelError`` when a download fails or does not match its pinned SHA-256.
    """
    progress = progress or NullProgress()
    wanted = list(models)
    missing = [m for m in wanted if model_status(m) != "ok"]
    if missing:
        progress.start("models", sum(m.size for m in missing), "bytes")
        try:
            for model in missing:
                download(model, model_path(model), progress)
        finally:
            progress.finish()
    return {m.name: model_path(m) for m in wanted}


def download(model: ModelFile, path: Path, progress: Progress | None = None) -> None:
    """Download ``model`` to ``path`` through a temporary file, checking size and SHA-256."""
    progress = progress or NullProgress()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(model.url, timeout=_TIMEOUT_SECONDS) as response,
            tmp.open("wb") as out,
        ):
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
                progress.advance(len(chunk))
                if size > model.size:
                    break
    except (urllib.error.URLError, OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        reason = getattr(exc, "reason", None) or exc
        raise ModelError(
            f"could not download {model.filename} ({reason}). tripsort needs it once: "
            "connect to the internet and run `tripsort models download`, or save the file "
            f"from {model.url} into {path.parent}"
        ) from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if size != model.size or digest.hexdigest() != model.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"the download of {model.filename} does not match its pinned SHA-256 "
            f"({size:,} bytes, expected {model.size:,}); it was not used. Try again later."
        )
    tmp.replace(path)
