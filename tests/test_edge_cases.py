"""Failure paths that need a simulated fault: locked files, read errors, odd sizes."""

import io
import zlib
from pathlib import Path

import pytest
from factories import jpeg, photo
from rich.console import Console
from typer.testing import CliRunner

import tripsort.cli as cli
import tripsort.export.writer as writer
import tripsort.ingest.exif as exif
import tripsort.ingest.pipeline as pipeline
from tripsort.api import Project
from tripsort.ingest.zip_reader import ZipSource
from tripsort.stats import count, human_bytes

KEEPER = "20260714_102345_mine_PXL_20260714_082345120.jpg"
COPY = "20260714_000000_sam_IMG-20260714-WA0001.jpg"


def test_read_errors_during_export_are_problems(
    project: Project, trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project.ingest([trip["sam"]])

    def broken(self: ZipSource, member: object, chunk_size: int = 0):  # type: ignore[no-untyped-def]
        yield b"partial"
        raise zlib.error("invalid stored block lengths")

    monkeypatch.setattr(ZipSource, "iter_chunks", broken)
    result = project.export()
    assert len(result.problems) == 3  # without mine.zip, the WhatsApp copy is a keeper too
    assert "invalid stored block lengths" in result.problems[0].message
    leftovers = list((project.folder / "all").glob("*.tripsort-tmp"))
    assert leftovers == []


def test_a_stale_file_that_is_already_gone(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    project.export(export_duplicates=True)
    folder = project.folder / "duplicates" / KEEPER.removesuffix(".jpg")
    (folder / COPY).unlink()
    result = project.export()
    assert (result.removed, result.left_alone) == (1, 0)
    assert not (project.folder / "duplicates").exists()


def test_a_locked_stale_file_is_retried_next_time(
    project: Project, trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    project.export(export_duplicates=True)
    real_unlink = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if "duplicates" in self.parts:
            raise PermissionError(32, "The process cannot access the file")  # Windows, file open
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    first = project.export()
    assert (first.removed, first.left_alone) == (0, 2)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    second = project.export()
    assert second.removed == 2
    assert not (project.folder / "duplicates").exists()


def test_a_link_whose_source_could_not_be_written(project: Project, trip: dict[str, Path]) -> None:
    project.ingest([trip["mine"], trip["sam"]])
    project.export()
    (project.folder / "all" / KEEPER).unlink()
    trip["mine"].unlink()  # the keeper cannot be extracted again
    result = project.export(export_duplicates=True)
    messages = [p.message for p in result.problems]
    assert any("mine.zip" in m for m in messages)
    assert any("cannot be linked" in m for m in messages)


def test_set_mtime_ignores_missing_and_impossible_dates(tmp_path: Path) -> None:
    path = tmp_path / "f.jpg"
    path.write_bytes(b"x")
    before = path.stat().st_mtime_ns
    writer._set_mtime(path, None)
    writer._set_mtime(path, "not a date")
    writer._set_mtime(path, "0001-01-01T00:00:00")
    assert path.stat().st_mtime_ns == before


def test_zip_entry_that_needs_a_missing_codec(
    project: Project, trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_codec(self: ZipSource, member: object, chunk_size: int = 0):  # type: ignore[no-untyped-def]
        raise NotImplementedError("That compression method is not supported")
        yield b""  # pragma: no cover

    monkeypatch.setattr(ZipSource, "iter_chunks", no_codec)
    result = project.ingest([trip["sam"]])
    assert result.sources[0].skipped == 3
    assert all("standard Deflate" in p.message for p in result.problems)


def test_images_too_large_to_buffer_are_passed_through(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from factories import make_zip

    monkeypatch.setattr(pipeline, "MAX_BUFFERED_IMAGE_BYTES", 10)
    project.ingest([make_zip(tmp_path / "big.zip", [("huge.jpg", jpeg(photo(1)))])])
    note, phash = project.connection.execute("SELECT note, phash FROM media").fetchone()
    assert phash is None
    assert "too large to decode" in note


def test_an_exif_parser_crash_means_no_exif(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    def crash(data: object) -> None:
        raise struct_error("unpack requires a buffer of 4 bytes")

    struct_error = type("error", (Exception,), {})
    monkeypatch.setattr(exif, "_read", crash)
    with Image.open(io.BytesIO(jpeg(photo(1, (32, 24))))) as image:
        assert exif.read_exif(image) == exif.NO_EXIF


@pytest.mark.parametrize(
    ("n", "text"),
    [(0, "0 B"), (999, "999 B"), (1500, "1.5 KB"), (2_500_000, "2.5 MB"), (3 * 10**9, "3.0 GB")],
)
def test_human_bytes(n: int, text: str) -> None:
    assert human_bytes(n) == text


def test_count() -> None:
    assert count(1, "file") == "1 file"
    assert count(1234, "file") == "1,234 files"
    assert count(2, "copy", "copies") == "2 copies"


def test_progress_bar_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(cli, "err", Console(file=out, force_terminal=True, width=100))
    progress = cli._RichProgress()
    progress.start("ingest", 1000, "bytes")
    progress.advance(500)
    progress.finish()
    progress.start("export", 3, "files")
    progress.advance(3)
    progress.finish()
    progress.start("export", 0, "files")  # nothing to do: no bar at all
    progress.advance(1)
    progress.finish()
    assert "ingest" in out.getvalue()


def test_os_errors_are_user_errors(
    project: Project, trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import tripsort.api as api

    project.ingest([trip["sam"]])
    folder = project.folder
    project.close()

    def denied(path: object) -> None:
        raise PermissionError(13, "Access is denied", str(path))

    monkeypatch.setattr(api.shutil, "rmtree", denied)
    result = CliRunner().invoke(cli.app, ["clean", str(folder), "--yes"])
    assert result.exit_code == 1
    assert "Access is denied" in result.output
