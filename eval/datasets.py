"""The private labeled set: friends' trip photos, kept outside the repository.

``TRIPSORT_EVAL_DIR`` names a folder with one subfolder of photos per source (per phone, or
per person who sent photos). The subfolders are zipped as ``A.zip``, ``B.zip``, ... in sorted
order, so nothing the index or the output prints carries their real names.

Nothing here prints a file or folder name: the photos and their names are friends'.
"""

import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tripsort.ingest.zip_reader import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

EVAL_DIR_ENV = "TRIPSORT_EVAL_DIR"


@dataclass(frozen=True)
class Source:
    label: str  # 'A', 'B', ...: the zip and index label
    folder: Path  # the real subfolder, used to map labels.csv rows to zip members
    zip_path: Path


def eval_dir() -> Path:
    """The labeled set's folder, from the environment or (on Windows) the user's registry,
    since a shell started before `setx` does not see the variable."""
    value = os.environ.get(EVAL_DIR_ENV)
    if not value and sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value = str(winreg.QueryValueEx(key, EVAL_DIR_ENV)[0])
        except OSError:
            value = None
    if not value:
        raise SystemExit(f"set {EVAL_DIR_ENV} to the folder holding the labeled photos")
    path = Path(value).expanduser()
    if not path.is_dir():
        raise SystemExit(f"{EVAL_DIR_ENV} does not name a folder")
    return path


def photo_files(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.relative_to(folder).as_posix(),
    )


def media_files(folder: Path) -> list[Path]:
    """Photos and videos: what the eval zips hold (videos for the video labels and the
    memory check; the photo evaluations ignore them)."""
    kinds = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in kinds),
        key=lambda p: p.relative_to(folder).as_posix(),
    )


def build_sources(root: Path, out: Path) -> list[Source]:
    """Zip each subfolder of ``root`` into ``out`` (stored, not compressed), reusing zips
    built earlier so their fingerprints, and the index, stay valid."""
    folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not folders and photo_files(root):
        folders = [root]  # a flat folder of photos is one source
    if not folders:
        raise SystemExit(
            "the labeled set has no photos (one subfolder per phone, or a flat folder)"
        )
    if len(folders) > 26:
        raise SystemExit("the labeled set has more than 26 subfolders")
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    for index, folder in enumerate(folders):
        label = chr(ord("A") + index)
        zip_path = out / f"{label}.zip"
        if not zip_path.exists():
            tmp = zip_path.with_suffix(".tmp")
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                for path in media_files(folder):
                    zf.write(path, path.relative_to(folder).as_posix())
            tmp.replace(zip_path)
        sources.append(Source(label, folder, zip_path))
    return sources


def file_key(source: Source, member: str) -> str:
    """How labels.csv names a photo: '<subfolder>/<path inside it>'."""
    return f"{source.folder.name}/{PurePosixPath(member).as_posix()}"
