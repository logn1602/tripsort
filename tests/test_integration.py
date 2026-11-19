"""The spec's integration test: zips built at test time, run through the CLI end to end.

* Two zips with exact duplicates across zips, a WhatsApp-style copy, and a HEIC file.
* `tripsort run`, then the folder tree, counts, keeper choice, and manifest fields.
* Run it again: nothing is decoded or written again.
* Add a third zip: only its new files are processed.
"""

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from factories import camera_exif, jpeg, make_zip, photo
from typer.testing import CliRunner

import tripsort.ingest.pipeline as pipeline
from tripsort.cli import app

runner = CliRunner()


@pytest.fixture
def decode_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = pipeline.analyze_image

    def counting(data: bytes, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(len(data))
        return real(data, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_image", counting)
    return calls


def run(*args: Path | str) -> int:
    result = runner.invoke(app, ["run", *map(str, args)])
    return result.exit_code


def tree(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        p.relative_to(folder).as_posix(): (p.stat().st_ino, p.stat().st_mtime_ns)
        for p in folder.rglob("*")
        if p.is_file() and ".tripsort" not in p.parts
    }


def manifest(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def test_trip_end_to_end(trip: dict[str, Path], tmp_path: Path, decode_calls: list[int]) -> None:
    out = tmp_path / "sorted"
    assert run(trip["mine"], trip["sam"], "-o", out, "--export-duplicates") == 2  # broken.jpg
    assert len(decode_calls) == 6  # every distinct image once, including broken.jpg

    files = tree(out)
    all_files = sorted(name for name in files if name.startswith("all/"))
    assert [name.split("/", 1)[1] for name in all_files] == [
        "20260714_101010_mine_VID_20260714_101010.mp4",
        "20260714_102345_mine_PXL_20260714_082345120.jpg",
        "20260714_110000_mine_PXL_20260714_090000000.jpg",
        "20260715_090000_mine_IMG_0001.HEIC",
        "20260716_120000_sam_IMG_20260716_120000.jpg",
    ]
    assert sum(name.startswith("unreadable/") for name in files) == 1
    assert sum(name.startswith("duplicates/") for name in files) == 2
    with zipfile.ZipFile(trip["mine"]) as zf:  # the bytes are the original's, untouched
        original = zf.read("DCIM/PXL_20260714_082345120.jpg")
    assert (out / all_files[1]).read_bytes() == original

    data = manifest(out)
    assert data["summary"]["exact_duplicates"] == 1
    assert data["summary"]["near_copies"] == 1
    keeper = next(f for f in data["files"] if f["paths"] and f["paths"][0] == all_files[1])
    copy = next(f for f in data["files"] if f["copy_of"])
    assert copy["copy_of"] == keeper["sha256"]  # the original won over the WhatsApp copy
    assert copy["occurrences"][0]["source"] == "sam"

    # The same command again: nothing is decoded, written, or rewritten.
    first_manifest = (out / "manifest.json").read_bytes()
    decode_calls.clear()
    assert run(trip["mine"], trip["sam"], "-o", out, "--export-duplicates") == 2
    assert decode_calls == []
    assert tree(out) == files
    assert (out / "manifest.json").read_bytes() == first_manifest

    # A third zip: only its new files are processed.
    with zipfile.ZipFile(trip["mine"]) as zf:
        heic = zf.read("DCIM/IMG_0001.HEIC")
    alex = make_zip(
        tmp_path / "in" / "alex.zip",
        [
            ("IMG_20260717_080000.jpg", jpeg(photo(9), exif=camera_exif("2026:07:17 08:00:00"))),
            ("copy_of_mine.heic", heic),
        ],
    )
    assert run(trip["mine"], trip["sam"], alex, "-o", out, "--export-duplicates") == 2
    assert len(decode_calls) == 1
    after = tree(out)
    assert sorted(set(after) - set(files)) == ["all/20260717_080000_alex_IMG_20260717_080000.jpg"]
    assert all(after[name] == files[name] for name in files if name != "manifest.json")
    assert manifest(out)["summary"]["exact_duplicates"] == 2
