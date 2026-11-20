import shutil
import zipfile
from pathlib import Path

import pytest
from factories import camera_exif, jpeg, make_zip, patch_entry, photo

import tripsort.ingest.pipeline as pipeline
from tripsort.api import Project
from tripsort.errors import ZipError


def count(project: Project, table: str) -> int:
    return int(project.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def reasons(project: Project) -> dict[str, str]:
    rows = project.connection.execute("SELECT member_path, reason FROM skipped_members")
    return {str(path): str(reason) for path, reason in rows}


def test_first_ingest(project: Project, trip: dict[str, Path]) -> None:
    result = project.ingest([trip["mine"], trip["sam"]])
    mine, sam = result.sources
    assert (mine.label, mine.status, mine.new_files, mine.known_files) == ("mine", "new", 5, 0)
    assert (mine.skipped, mine.unreadable) == (3, 1)
    assert (sam.label, sam.new_files, sam.known_files) == ("sam", 2, 1)  # one exact duplicate
    assert result.decoded == 5  # 3 images in mine, 2 in sam; not the video or broken.jpg
    assert count(project, "media") == 7
    assert count(project, "occurrences") == 8
    assert reasons(project) == {
        "__MACOSX/DCIM/._PXL_20260714_082345120.jpg": "junk",
        ".DS_Store": "junk",
        "notes.txt": "unsupported_type",
    }
    assert [p.member for p in result.problems] == ["DCIM/broken.jpg"]
    assert "unreadable/" in result.problems[0].message
    statuses = project.connection.execute("SELECT status, COUNT(*) FROM sources GROUP BY status")
    assert dict(statuses.fetchall()) == {"complete": 2}


def test_media_rows(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"]])
    rows = {
        row["member_path"]: row
        for row in project.connection.execute(
            "SELECT o.member_path, m.* FROM occurrences o JOIN media m ON m.id = o.media_id"
        )
    }
    rotated = rows["DCIM/PXL_20260714_090000000.jpg"]
    assert (rotated["width"], rotated["height"]) == (900, 1200)  # orientation 6
    assert rotated["has_exif"] == 1
    assert rotated["camera_model"] == "Pixel 7"
    assert rotated["exif_datetime"] == "2026-07-14T11:00:00"
    assert rotated["exif_modified"] == "2026-07-14T11:00:00"
    assert (rotated["exposure_time"], rotated["iso"]) == (pytest.approx(1 / 120), 100)
    heic = rows["DCIM/IMG_0001.HEIC"]
    assert (heic["format"], heic["camera_model"]) == ("HEIF", "iPhone 15")
    video = rows["DCIM/VID_20260714_101010.mp4"]
    assert (video["kind"], video["format"], video["phash"]) == ("video", "MP4", None)
    assert "exact duplicates only" in video["note"]
    broken = rows["DCIM/broken.jpg"]
    assert broken["status"] == "unreadable"
    assert "truncated" in broken["note"]


def test_second_ingest_reads_nothing(
    project: Project, trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project.ingest([trip["mine"], trip["sam"]])

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("a known zip was read again")

    monkeypatch.setattr(pipeline.ZipSource, "iter_chunks", fail)
    result = project.ingest([trip["mine"], trip["sam"]])
    assert [s.status for s in result.sources] == ["already indexed", "already indexed"]
    assert result.decoded == 0
    assert result.files_read == 0
    assert count(project, "occurrences") == 8
    assert "already indexed, nothing to read" in result.summary()


def test_same_zip_twice_in_one_call(project: Project, trip: dict[str, Path]) -> None:
    result = project.ingest([trip["sam"], trip["sam"]])
    assert len(result.sources) == 1


def test_new_zip_only_processes_its_new_files(
    project: Project, trip: dict[str, Path], tmp_path: Path
) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    with zipfile.ZipFile(trip["mine"]) as zf:
        known = zf.read("DCIM/IMG_0001.HEIC")
    alex = make_zip(
        tmp_path / "in" / "alex.zip",
        [("IMG_20260717_080000.jpg", jpeg(photo(9))), ("from_mine.heic", known)],
    )
    result = project.ingest([trip["mine"], trip["sam"], alex])
    assert result.decoded == 1
    assert (result.sources[2].new_files, result.sources[2].known_files) == (1, 1)


def test_moved_zip_is_recognized(project: Project, trip: dict[str, Path], tmp_path: Path) -> None:
    project.ingest([trip["sam"]])
    moved = tmp_path / "elsewhere" / "sam-renamed.zip"
    moved.parent.mkdir()
    shutil.move(trip["sam"], moved)
    result = project.ingest([moved])
    assert result.sources[0].status == "already indexed"
    assert result.sources[0].label == "sam"  # labels stick
    row = project.connection.execute("SELECT zip_path, zip_name FROM sources").fetchone()
    assert (row[0], row[1]) == (str(moved.resolve()), "sam-renamed.zip")


def test_interrupted_ingest_resumes(
    project: Project, trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pipeline.analyze_image
    calls = {"n": 0}

    def interrupt_on_third(data: bytes, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(data, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_image", interrupt_on_third)
    with pytest.raises(KeyboardInterrupt):
        pipeline.ingest_zips(project.connection, [trip["mine"]], batch_size=2)
    committed = count(project, "occurrences") + count(project, "skipped_members")
    assert 0 < committed < 8  # some batches made it, the interrupted one did not
    status = project.connection.execute("SELECT status FROM sources").fetchone()[0]
    assert status == "partial"

    monkeypatch.setattr(pipeline, "analyze_image", real)
    result = project.ingest([trip["mine"]])
    assert result.sources[0].status == "resumed"
    assert count(project, "occurrences") + count(project, "skipped_members") == 8
    assert count(project, "media") == 5
    assert project.connection.execute("SELECT status FROM sources").fetchone()[0] == "complete"


def test_entries_that_cannot_be_read_are_reported_with_a_hint(
    project: Project, tmp_path: Path
) -> None:
    good = jpeg(photo(1))
    data = make_zip(
        tmp_path / "raw.zip",
        [
            ("locked.jpg", good),
            ("deflate64.jpg", good),
            ("crc.jpg", jpeg(photo(2))),
            ("ok.jpg", good),
        ],
    ).read_bytes()
    data = patch_entry(data, "locked.jpg", set_flags=0x1)
    data = patch_entry(data, "deflate64.jpg", method=9)
    data = patch_entry(data, "crc.jpg", corrupt=True)
    path = tmp_path / "broken.zip"
    path.write_bytes(data)
    result = project.ingest([path])
    assert reasons(project) == {
        "locked.jpg": "encrypted",
        "deflate64.jpg": "unsupported_compression",
        "crc.jpg": "corrupt",
    }
    messages = {p.member: p.message for p in result.problems}
    assert "without encryption" in messages["locked.jpg"]
    assert "Deflate64" in messages["deflate64.jpg"]
    assert "corrupt" in messages["crc.jpg"]
    assert "Download or create the zip again" in messages["crc.jpg"]
    assert count(project, "media") == 1


def test_labels_are_unique(project: Project, tmp_path: Path) -> None:
    a = make_zip(tmp_path / "a" / "photos.zip", [("1.jpg", jpeg(photo(1)))])
    b = make_zip(tmp_path / "b" / "photos.zip", [("2.jpg", jpeg(photo(2)))])
    result = project.ingest([a, b])
    assert [s.label for s in result.sources] == ["photos", "photos-2"]


def test_bad_paths_fail_before_any_work(
    project: Project, trip: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ZipError, match="no such file"):
        project.ingest([trip["mine"], tmp_path / "missing.zip"])
    assert count(project, "sources") == 0
    not_zip = tmp_path / "fake.zip"
    not_zip.write_text("hello")
    with pytest.raises(ZipError, match="not a readable zip"):
        project.ingest([not_zip])


def test_exact_duplicates_inside_one_zip(project: Project, tmp_path: Path) -> None:
    data = jpeg(photo(1), exif=camera_exif())
    path = make_zip(tmp_path / "z.zip", [("DCIM/IMG_1.jpg", data), ("Backup/IMG_1.jpg", data)])
    result = project.ingest([path])
    assert (result.sources[0].new_files, result.sources[0].known_files) == (1, 1)
    assert count(project, "media") == 1
    assert count(project, "occurrences") == 2
