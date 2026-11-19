from pathlib import Path

import pytest
from factories import camera_exif, jpeg, make_zip, photo

import tripsort
from tripsort.api import Project, clean
from tripsort.errors import UserError


def test_package_exposes_the_api_lazily() -> None:
    assert tripsort.Project is Project
    assert tripsort.clean is clean
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = tripsort.nope  # type: ignore[attr-defined]


def test_open_creates_the_folder_and_index(tmp_path: Path) -> None:
    with Project.open(tmp_path / "new" / "sorted") as project:
        assert project.folder == (tmp_path / "new" / "sorted").resolve()
        assert (project.folder / ".tripsort" / "index.db").is_file()


def test_open_refuses_a_file(tmp_path: Path) -> None:
    target = tmp_path / "x"
    target.write_text("not a folder")
    with pytest.raises(UserError, match="is not a folder"):
        Project.open(target)


def test_analysis_goes_stale_when_the_index_or_config_changes(
    project: Project, trip: dict[str, Path]
) -> None:
    assert project.analysis_is_stale()
    project.ingest([trip["mine"]])
    project.analyze()
    assert not project.analysis_is_stale()
    project.ingest([trip["sam"]])
    assert project.analysis_is_stale()
    project.analyze()
    (project.folder / "tripsort.toml").write_text("[thresholds]\nphash_max_distance = 0\n")
    folder = project.folder
    project.close()
    with Project.open(folder) as reopened:
        assert reopened.analysis_is_stale()


def test_export_analyzes_first_when_needed(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    result = project.export()  # no explicit analyze()
    assert result.near_copies == 1
    assert not project.analysis_is_stale()


def test_analyze_summary(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    result = project.analyze()
    summary = result.summary()
    assert "8 files in 2 zips: 7 unique, 1 exact duplicate." in summary
    assert "1 recompressed copy of 1 photo" in summary
    assert "1 file (videos, very large images) checked for exact duplicates only." in summary
    assert "1 file could not be decoded" in summary
    assert result.dates == {"exif": 4, "filename": 2, "zip_mtime": 1}  # the video is named by date


def test_an_unedited_original_is_kept_over_a_larger_edited_save(
    project: Project, tmp_path: Path
) -> None:
    original = jpeg(photo(1), quality=80, exif=camera_exif())
    edited = jpeg(photo(1), quality=95, exif=camera_exif(modified="2026:11:02 18:05:12"))
    assert len(edited) > len(original)
    project.ingest([make_zip(tmp_path / "trip.zip", [("a.jpg", edited), ("b.jpg", original)])])
    assert project.analyze().stats.near_copies == 1
    keepers = project.connection.execute(
        "SELECT o.member_path FROM group_members g JOIN occurrences o ON o.media_id = g.media_id "
        "WHERE g.is_keeper = 1"
    )
    assert [row[0] for row in keepers] == ["b.jpg"]


def test_runs_are_recorded_with_peak_memory(project: Project, trip: dict[str, Path]) -> None:
    ingested = project.ingest([trip["sam"]])
    analyzed = project.analyze()
    exported = project.export()
    rows = project.connection.execute(
        "SELECT stage, items, seconds, peak_rss_bytes, finished_at FROM runs ORDER BY id"
    ).fetchall()
    assert [row[0] for row in rows] == ["ingest", "analyze", "export"]
    assert all(row[3] and row[3] > 0 and row[4] for row in rows)
    assert ingested.peak_rss_bytes
    assert analyzed.peak_rss_bytes
    assert exported.peak_rss_bytes


def test_clean(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["sam"]])
    project.export()
    folder = project.folder
    project.close()
    assert clean(folder) is True
    assert not (folder / ".tripsort").exists()
    assert (folder / "all").is_dir()
    assert clean(folder) is False
