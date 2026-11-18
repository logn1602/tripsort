import json
from pathlib import Path
from typing import Any

from factories import camera_exif, jpeg, make_zip, patch_entry, photo

from tripsort import __version__
from tripsort.api import Project

FILE_FIELDS = {
    "sha256": str,
    "kind": str,
    "format": (str, type(None)),
    "bytes": int,
    "width": (int, type(None)),
    "height": (int, type(None)),
    "taken_at": (str, type(None)),
    "tz_offset": (str, type(None)),
    "date_source": (str, type(None)),
    "camera": (dict, type(None)),
    "phash": (str, type(None)),
    "status": str,
    "note": (str, type(None)),
    "keeper": bool,
    "copy_of": (str, type(None)),
    "copy_distance": (int, type(None)),
    "paths": list,
    "occurrences": list,
}
TOP_LEVEL = {
    "manifest_version",
    "tripsort_version",
    "config_hash",
    "thresholds",
    "analyzers",
    "summary",
    "sources",
    "files",
    "skipped",
}


def load(project: Project) -> dict[str, Any]:
    return json.loads((project.folder / "manifest.json").read_text(encoding="utf-8"))


def by_member(manifest: dict[str, Any], member: str) -> dict[str, Any]:
    for entry in manifest["files"]:
        if any(o["member"] == member for o in entry["occurrences"]):
            return entry
    raise KeyError(member)


def test_schema(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    project.export(export_duplicates=True)
    manifest = load(project)
    assert set(manifest) == TOP_LEVEL
    assert manifest["manifest_version"] == 1
    assert manifest["tripsort_version"] == __version__
    assert manifest["thresholds"] == {
        "phash_max_distance": 6,
        "phash_max_distance_both_exif": 2,
        "copy_max_aspect_diff": 0.02,
    }
    for entry in manifest["files"]:
        assert set(entry) == set(FILE_FIELDS)
        for key, kind in FILE_FIELDS.items():
            assert isinstance(entry[key], kind), (key, entry[key])
        for occurrence in entry["occurrences"]:
            assert set(occurrence) == {"source", "member", "member_raw"}
    for source in manifest["sources"]:
        assert set(source) == {"label", "zip", "fingerprint", "members", "status"}
    dates = [entry["taken_at"] or "" for entry in manifest["files"]]
    assert dates == sorted(dates)


def test_contents(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    project.export(export_duplicates=True)
    manifest = load(project)
    summary = manifest["summary"]
    assert (summary["files_found"], summary["unique"], summary["exact_duplicates"]) == (8, 7, 1)
    assert (summary["near_copies"], summary["kept"], summary["unreadable"]) == (1, 5, 1)
    assert summary["skipped"] == {"junk": 2, "unsupported_type": 1}
    assert summary["bytes_saved"] == summary["bytes_in"] - summary["bytes_kept"] > 0

    original = by_member(manifest, "DCIM/PXL_20260714_082345120.jpg")
    copy = by_member(manifest, "WhatsApp Images/IMG-20260714-WA0001.jpg")
    assert original["keeper"] is True
    assert (copy["keeper"], copy["copy_of"]) == (False, original["sha256"])
    assert copy["copy_distance"] <= 6
    assert copy["date_source"] == "filename"
    assert original["date_source"] == "exif"
    assert original["taken_at"] == "2026-07-14T10:23:45.120"
    assert original["camera"] == {"make": "Google", "model": "Pixel 7", "lens": None}
    assert len(original["paths"]) == 2  # all/ and the duplicates/ review folder

    duplicate = by_member(manifest, "Camera/same_as_mine.jpg")
    assert [o["source"] for o in duplicate["occurrences"]] == ["mine", "sam"]

    broken = by_member(manifest, "DCIM/broken.jpg")
    assert broken["status"] == "unreadable"
    assert broken["paths"][0].startswith("unreadable/")

    assert [(s["member"], s["reason"]) for s in manifest["skipped"]] == [
        ("notes.txt", "unsupported_type")
    ]  # junk is only counted


def test_no_absolute_paths_and_no_gps(project: Project, tmp_path: Path) -> None:
    with_gps = jpeg(photo(1), exif=camera_exif(gps=(48.858411, 2.294507)))
    zip_path = make_zip(tmp_path / "secret-folder" / "mine.zip", [("IMG_1.jpg", with_gps)])
    project.ingest([zip_path])
    project.export()
    text = (project.folder / "manifest.json").read_text(encoding="utf-8")
    assert "secret-folder" not in text
    assert str(tmp_path) not in text
    assert "48.85" not in text
    assert "gps" not in text.lower()
    gps = project.connection.execute("SELECT gps_lat FROM media").fetchone()[0]
    assert abs(gps - 48.858411) < 1e-4  # it is kept in the local index only


def test_raw_names_and_read_failures(project: Project, tmp_path: Path) -> None:
    raw = "é.jpg".encode("cp437")
    data = make_zip(
        tmp_path / "raw.zip", [("x" * len(raw), jpeg(photo(1))), ("locked.jpg", jpeg(photo(2)))]
    ).read_bytes()
    data = patch_entry(data, "x" * len(raw), new_name=raw)
    data = patch_entry(data, "locked.jpg", set_flags=0x1)
    path = tmp_path / "old.zip"
    path.write_bytes(data)
    project.ingest([path])
    project.export()
    manifest = load(project)
    assert manifest["files"][0]["occurrences"][0] == {
        "source": "old",
        "member": "é.jpg",
        "member_raw": raw.hex(),
    }
    (locked,) = manifest["skipped"]
    assert (locked["member"], locked["reason"]) == ("locked.jpg", "encrypted")
    assert "without encryption" in locked["hint"]


def test_same_inputs_same_bytes(project: Project, trip: dict[str, Path], tmp_path: Path) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    project.export()
    first = (project.folder / "manifest.json").read_bytes()
    with Project.open(tmp_path / "other") as other:
        other.ingest([trip["sam"], trip["mine"]])  # zips in the other order
        other.export()
        second = (other.folder / "manifest.json").read_bytes()
    assert first == second
