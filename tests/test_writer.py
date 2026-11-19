import shutil
from datetime import datetime
from pathlib import Path

import pytest
from factories import camera_exif, jpeg, make_zip, photo, whatsapp_copy

import tripsort.export.writer as writer
from tripsort.api import Project, clean
from tripsort.errors import ExportError

KEEPER = "20260714_102345_mine_PXL_20260714_082345120.jpg"
COPY = "20260714_000000_sam_IMG-20260714-WA0001.jpg"


def files(folder: Path) -> list[str]:
    return sorted(
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and ".tripsort" not in p.parts
    )


def snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        rel: (st.st_ino, st.st_mtime_ns) for rel in files(folder) if (st := (folder / rel).stat())
    }


@pytest.fixture
def ingested(project: Project, trip: dict[str, Path]) -> Project:
    project.ingest([trip["mine"], trip["sam"]])
    project.analyze()
    return project


def test_first_export(ingested: Project) -> None:
    result = ingested.export()
    assert files(ingested.folder / "all") == [
        "20260714_101010_mine_VID_20260714_101010.mp4",
        KEEPER,
        "20260714_110000_mine_PXL_20260714_090000000.jpg",
        "20260715_090000_mine_IMG_0001.HEIC",
        "20260716_120000_sam_IMG_20260716_120000.jpg",
    ]
    unreadable = files(ingested.folder / "unreadable")
    assert len(unreadable) == 1
    assert unreadable[0].endswith("_mine_broken.jpg")
    assert (ingested.folder / "manifest.json").is_file()
    assert (result.written, result.kept, result.unreadable, result.near_copies) == (6, 5, 1, 1)
    assert result.manifest == ingested.folder / "manifest.json"
    assert "1 near-copy was left out of all/" in result.summary()
    assert "--export-duplicates" in result.summary()


def test_files_get_the_photo_date_as_mtime(ingested: Project) -> None:
    ingested.export()
    expected = datetime(2026, 7, 14, 10, 23, 45, 120000).timestamp()
    assert (ingested.folder / "all" / KEEPER).stat().st_mtime == pytest.approx(expected, abs=2)


def test_second_export_writes_nothing(ingested: Project) -> None:
    ingested.export(export_duplicates=True)
    before = snapshot(ingested.folder)
    result = ingested.export(export_duplicates=True)
    assert (result.written, result.removed) == (0, 0)
    assert result.unchanged == result.planned
    assert snapshot(ingested.folder) == before


def test_dry_run_writes_nothing(ingested: Project) -> None:
    result = ingested.export(dry_run=True, export_duplicates=True)
    assert result.dry_run
    assert result.written == 8
    assert result.bytes_written > 0
    assert files(ingested.folder) == []
    assert result.manifest is None
    assert "Would write 8 files" in result.summary()
    count = ingested.connection.execute("SELECT COUNT(*) FROM exported_files").fetchone()[0]
    assert count == 0


def test_export_duplicates_links_the_keeper(ingested: Project) -> None:
    result = ingested.export(export_duplicates=True)
    folder = ingested.folder / "duplicates" / KEEPER.removesuffix(".jpg")
    assert sorted(p.name for p in folder.iterdir()) == [COPY, KEEPER]
    assert (folder / KEEPER).samefile(ingested.folder / "all" / KEEPER)
    assert result.linked == 1
    assert result.link_fallbacks == 0
    assert "duplicates/ shows each one next to the file that was kept" in result.summary()


def test_copies_when_the_drive_refuses_hardlinks(
    ingested: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: Path, target: Path) -> None:
        raise OSError(1, "Incorrect function")  # what exFAT says on Windows

    monkeypatch.setattr(Path, "hardlink_to", refuse)
    result = ingested.export(export_duplicates=True)
    folder = ingested.folder / "duplicates" / KEEPER.removesuffix(".jpg")
    assert (folder / KEEPER).read_bytes() == (ingested.folder / "all" / KEEPER).read_bytes()
    assert not (folder / KEEPER).samefile(ingested.folder / "all" / KEEPER)
    assert (result.linked, result.link_fallbacks) == (0, 1)
    assert "does not support hardlinks" in result.summary()


def test_copy_option_never_links(ingested: Project) -> None:
    result = ingested.export(export_duplicates=True, hardlinks=False)
    folder = ingested.folder / "duplicates" / KEEPER.removesuffix(".jpg")
    assert not (folder / KEEPER).samefile(ingested.folder / "all" / KEEPER)
    assert (result.linked, result.link_fallbacks) == (0, 0)


def test_dropping_export_duplicates_removes_the_review_folder(ingested: Project) -> None:
    ingested.export(export_duplicates=True)
    result = ingested.export()
    assert result.removed == 2
    assert not (ingested.folder / "duplicates").exists()


def test_files_edited_after_export_are_left_alone(ingested: Project) -> None:
    ingested.export(export_duplicates=True)
    copy = ingested.folder / "duplicates" / KEEPER.removesuffix(".jpg") / COPY
    copy.write_bytes(b"edited by the user")
    result = ingested.export()
    assert copy.read_bytes() == b"edited by the user"
    assert (result.removed, result.left_alone) == (1, 1)
    assert "changed after the last export" in result.summary()
    again = ingested.export()  # the file is no longer tripsort's: never touched again
    assert (again.removed, again.left_alone) == (0, 0)
    assert copy.read_bytes() == b"edited by the user"


def test_edited_file_in_all_is_not_overwritten(ingested: Project) -> None:
    ingested.export()
    target = ingested.folder / "all" / KEEPER
    target.write_bytes(b"rotated in an editor")
    result = ingested.export()
    assert target.read_bytes() == b"rotated in an editor"
    assert result.left_alone == 1


def test_unknown_file_at_a_target_name_is_never_overwritten(ingested: Project) -> None:
    (ingested.folder / "all").mkdir()
    (ingested.folder / "all" / KEEPER).write_bytes(b"someone else's file")
    ingested.export()
    assert (ingested.folder / "all" / KEEPER).read_bytes() == b"someone else's file"
    assert (ingested.folder / "all" / KEEPER.replace(".jpg", "_1.jpg")).stat().st_size > 1000


def test_identical_files_are_adopted_after_clean(ingested: Project, trip: dict[str, Path]) -> None:
    ingested.export()
    before = snapshot(ingested.folder)
    folder = ingested.folder
    ingested.close()
    assert clean(folder)
    with Project.open(folder) as fresh:
        fresh.ingest([trip["mine"], trip["sam"]])
        result = fresh.export()
        assert result.written == 0
        assert result.unchanged == result.planned
    assert snapshot(folder) == before


def test_deleted_files_are_written_again(ingested: Project) -> None:
    ingested.export()
    (ingested.folder / "all" / KEEPER).unlink()
    result = ingested.export()
    assert result.written == 1
    assert (ingested.folder / "all" / KEEPER).is_file()


def test_a_better_original_replaces_its_copy(project: Project, tmp_path: Path) -> None:
    original = jpeg(photo(1), exif=camera_exif())
    first = make_zip(tmp_path / "sam.zip", [("IMG-20260714-WA0001.jpg", whatsapp_copy(original))])
    project.ingest([first])
    project.export()
    assert files(project.folder / "all") == [COPY]
    second = make_zip(tmp_path / "mine.zip", [("PXL_20260714_082345120.jpg", original)])
    project.ingest([second])
    result = project.export()
    assert files(project.folder / "all") == ["20260714_102345_mine_PXL_20260714_082345120.jpg"]
    assert (result.written, result.removed) == (1, 1)


def test_names_that_differ_only_in_case_get_a_suffix(project: Project, tmp_path: Path) -> None:
    zip_path = make_zip(
        tmp_path / "z.zip", [("a/IMG_1.JPG", jpeg(photo(1))), ("b/img_1.jpg", jpeg(photo(2)))]
    )
    project.ingest([zip_path])
    project.export()
    names = [name.split("_z_", 1)[1] for name in files(project.folder / "all")]
    assert sorted(names) == ["IMG_1.JPG", "img_1_1.jpg"] or sorted(names) == [
        "IMG_1_1.JPG",
        "img_1.jpg",
    ]


def test_missing_zip_is_reported(ingested: Project, trip: dict[str, Path]) -> None:
    trip["sam"].unlink()
    result = ingested.export()
    assert len(result.problems) == 1
    assert "sam.zip" in result.problems[0].message
    assert "tripsort ingest" in result.problems[0].message
    assert "could not be exported" in result.summary()


def test_changed_zip_is_reported(ingested: Project, trip: dict[str, Path]) -> None:
    make_zip(trip["sam"], [("other.jpg", jpeg(photo(8)))])  # same path, different zip
    result = ingested.export()
    assert len(result.problems) == 1


def test_not_enough_disk_space(ingested: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = shutil.disk_usage(ingested.folder)
    monkeypatch.setattr(writer.shutil, "disk_usage", lambda path: usage._replace(free=1024))
    with pytest.raises(ExportError, match=r"only 1\.0 KB is free"):
        ingested.export()
    assert not (ingested.folder / "all").exists()


def test_interrupted_export_recovers(ingested: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    real = writer._set_mtime
    calls = {"n": 0}

    def interrupt_on_third(path: Path, taken_at: str | None) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        real(path, taken_at)

    monkeypatch.setattr(writer, "_set_mtime", interrupt_on_third)
    with pytest.raises(KeyboardInterrupt):
        ingested.export()
    leftovers = [p for p in (ingested.folder / "all").iterdir() if p.name.endswith(".tripsort-tmp")]
    assert len(leftovers) == 1  # the file being written stays under its temporary name
    monkeypatch.setattr(writer, "_set_mtime", real)
    result = ingested.export()
    assert result.written == 4  # the two finished files were recorded before the interrupt
    assert not [p for p in (ingested.folder / "all").iterdir() if p.name.endswith(".tripsort-tmp")]
    assert len(files(ingested.folder / "all")) == 5
