import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from factories import camera_exif, jpeg, make_zip, patch_entry, photo
from typer.testing import CliRunner

from tripsort.cli import app

runner = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI colors, which Typer forces on when it detects CI."""
    return ANSI_ESCAPE.sub("", text)


def invoke(*args: str | Path, input: str | None = None):  # type: ignore[no-untyped-def]
    return runner.invoke(app, [str(a) for a in args], input=input)


@pytest.fixture
def clean_zip(tmp_path: Path) -> Path:
    return make_zip(
        tmp_path / "in" / "alex.zip",
        [
            ("IMG_20260714_102345.jpg", jpeg(photo(1), exif=camera_exif())),
            ("IMG_20260714_110000.jpg", jpeg(photo(2), exif=camera_exif("2026:07:14 11:00:00"))),
        ],
    )


def test_version_prints_package_version() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.output.strip() == f"tripsort {version('tripsort')}"


def test_no_arguments_shows_help() -> None:
    result = invoke()
    output = plain(result.output)
    assert "Usage" in output
    for command in ("run", "ingest", "analyze", "export", "clean"):
        assert command in output


def test_run_help_lists_arguments_and_options() -> None:
    result = invoke("run", "--help")
    assert result.exit_code == 0
    output = plain(result.output)
    for expected in (
        "ZIP",
        "--out",
        "-o",
        "--refs",
        "--copy",
        "--dry-run",
        "--export-duplicates",
        "--benchmark",
    ):
        assert expected in output


@pytest.mark.parametrize(
    "args",
    [("run", "mine.zip"), ("run", "mine.zip", "-o", "out", "--bogus"), ("export",), ("nope",)],
)
def test_usage_errors_exit_with_1(args: tuple[str, ...]) -> None:
    # Typer would exit with 2, which tripsort keeps for partial success.
    assert invoke(*args).exit_code == 1


def test_python_dash_m_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tripsort", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"tripsort {version('tripsort')}"


def test_run_clean_trip_exits_0(clean_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    result = invoke("run", clean_zip, "-o", out)
    assert result.exit_code == 0, result.output
    assert len(list((out / "all").iterdir())) == 2
    assert (out / "manifest.json").is_file()
    assert "2 files in 1 zip" in result.output


def test_run_with_problems_exits_2(trip: dict[str, Path], tmp_path: Path) -> None:
    result = invoke("run", trip["mine"], trip["sam"], "-o", tmp_path / "sorted", "--benchmark")
    assert result.exit_code == 2
    output = plain(result.output)
    assert "could not decode it" in output  # the truncated JPEG
    assert "1 near-copy was left out of all/" in output
    assert "--export-duplicates" in output
    assert "peak memory:" in output
    for stage in ("ingest", "analyze", "export"):
        assert re.search(rf"^\s+{stage}\s+[\d,]+ files", output, re.MULTILINE)


def test_unreadable_zip_entries_exit_2_with_advice(tmp_path: Path) -> None:
    data = make_zip(tmp_path / "raw.zip", [("locked.jpg", jpeg(photo(1)))]).read_bytes()
    locked = tmp_path / "locked.zip"
    locked.write_bytes(patch_entry(data, "locked.jpg", set_flags=0x1))
    result = invoke("run", locked, "-o", tmp_path / "sorted")
    assert result.exit_code == 2
    assert "without encryption" in plain(result.output)


def test_messages_keep_square_brackets(tmp_path: Path) -> None:
    """Rich reads [video] as a style: names and messages are escaped before printing."""
    data = make_zip(tmp_path / "raw.zip", [("IMG[copy].jpg", jpeg(photo(1)))]).read_bytes()
    locked = tmp_path / "locked.zip"
    locked.write_bytes(patch_entry(data, "IMG[copy].jpg", set_flags=0x1))
    result = invoke("run", locked, "-o", tmp_path / "sorted")
    assert result.exit_code == 2
    assert "IMG[copy].jpg" in plain(result.output)


def test_refs_with_no_photos_names_nobody(clean_zip: Path, tmp_path: Path) -> None:
    (tmp_path / "refs").mkdir()
    result = invoke("run", clean_zip, "-o", tmp_path / "sorted", "--refs", tmp_path / "refs")
    assert result.exit_code == 0
    assert "refs/ named 0 of 0 people" in plain(result.output)


def test_missing_refs_folder_fails_before_any_work(clean_zip: Path, tmp_path: Path) -> None:
    result = invoke("run", clean_zip, "-o", tmp_path / "sorted", "--refs", tmp_path / "nope")
    assert result.exit_code == 1
    assert "no such folder" in plain(result.output)
    assert not (tmp_path / "sorted").exists()


def test_missing_zip_is_a_user_error(tmp_path: Path) -> None:
    result = invoke("run", tmp_path / "missing.zip", "-o", tmp_path / "sorted")
    assert result.exit_code == 1
    assert "no such file" in plain(result.output)


def test_output_that_is_a_file_is_a_user_error(clean_zip: Path, tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")
    result = invoke("run", clean_zip, "-o", target)
    assert result.exit_code == 1
    assert "is not a folder" in plain(result.output)


def test_bad_config_is_a_user_error(clean_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    out.mkdir()
    (out / "tripsort.toml").write_text("[thresholds]\ntypo = 1\n", encoding="utf-8")
    result = invoke("run", clean_zip, "-o", out)
    assert result.exit_code == 1
    assert "unknown key 'typo'" in plain(result.output)


def test_step_by_step_commands(trip: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    ingested = invoke("ingest", out, trip["mine"], "--benchmark")
    assert ingested.exit_code == 2  # broken.jpg
    assert "mine.zip (mine): read" in ingested.output
    assert invoke("ingest", out, trip["sam"]).exit_code == 0
    analyzed = invoke("analyze", out, "--benchmark")
    assert analyzed.exit_code == 0
    assert "recompressed copy of 1 photo" in analyzed.output
    dry = invoke("export", out, "--dry-run")
    assert dry.exit_code == 2
    assert "Would write" in dry.output
    assert not (out / "all").exists()
    exported = invoke("export", out, "--export-duplicates", "--copy", "--benchmark")
    assert exported.exit_code == 2
    assert (out / "duplicates").is_dir()
    again = invoke("ingest", out, trip["mine"], trip["sam"])
    assert "already indexed, nothing to read" in again.output
    assert again.exit_code == 0  # nothing went wrong in this ingest


def test_export_choices_stick_until_turned_off(trip: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    duplicates = out / "duplicates"

    def linked() -> bool:
        """Whether the keeper in duplicates/ is a hardlink into all/ (else a copy)."""
        (folder,) = duplicates.iterdir()
        keeper = folder / f"{folder.name}.jpg"
        return keeper.samefile(out / "all" / keeper.name)

    first = invoke("run", trip["mine"], trip["sam"], "-o", out, "--copy", "--export-duplicates")
    assert first.exit_code == 2  # broken.jpg
    assert "Kept from the last export" not in first.output  # given, not kept
    assert not linked()

    both = (
        "Kept from the last export: --copy and --export-duplicates "
        "(--no-copy and --no-export-duplicates turn them off)."
    )
    copies = "Kept from the last export: --copy (--no-copy turns it off)."
    assert both in plain(invoke("export", out).output)
    dry = plain(invoke("export", out, "--dry-run", "--no-export-duplicates").output)
    assert copies in dry
    assert "2 to remove" in dry
    assert both in plain(invoke("export", out).output)  # the dry run saved nothing

    dropped = plain(invoke("export", out, "--no-export-duplicates").output)
    assert copies in dropped
    assert "2 removed" in dropped
    assert not duplicates.exists()
    assert copies in plain(invoke("export", out).output)  # --no-export-duplicates sticks
    assert not duplicates.exists()

    back = plain(invoke("export", out, "--no-copy", "--export-duplicates").output)
    assert "Kept from the last export" not in back
    assert linked()
    later = plain(invoke("export", out).output)
    assert "Kept from the last export: --export-duplicates (" in later  # --no-copy sticks

    for args in (
        ["export", out, "--copy", "--no-copy"],
        ["run", trip["mine"], "-o", out, "--export-duplicates", "--no-export-duplicates"],
    ):
        clash = invoke(*args)
        assert clash.exit_code == 1
        assert "cannot be used together" in plain(clash.output)


def test_clean(clean_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    assert invoke("run", clean_zip, "-o", out).exit_code == 0
    declined = invoke("clean", out, input="n\n")
    assert declined.exit_code == 1
    assert (out / ".tripsort").exists()
    accepted = invoke("clean", out, input="y\n")
    assert accepted.exit_code == 0
    assert not (out / ".tripsort").exists()
    assert (out / "all").is_dir()  # exported folders stay
    assert "Nothing to clean" in invoke("clean", out, "--yes").output


def test_ingest_without_zips_only_catches_up(clean_zip: Path, tmp_path: Path) -> None:
    out = tmp_path / "sorted"
    assert invoke("run", clean_zip, "-o", out).exit_code == 0
    result = invoke("ingest", out)
    assert result.exit_code == 0
    assert "wait for a face scan" not in plain(result.output)
