import re
import subprocess
import sys
from importlib.metadata import version

from typer.testing import CliRunner

from tripsort.cli import app

runner = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI colors, which Typer forces on when it detects CI."""
    return ANSI_ESCAPE.sub("", text)


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"tripsort {version('tripsort')}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    output = plain(result.output)
    assert "Usage" in output
    assert "run" in output


def test_run_help_lists_arguments_and_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    output = plain(result.output)
    for expected in ("ZIP", "--out", "-o", "--refs"):
        assert expected in output


def test_run_requires_out() -> None:
    result = runner.invoke(app, ["run", "mine.zip"])
    assert result.exit_code == 2
    assert "--out" in plain(result.output)


def test_run_stub_exits_with_user_error() -> None:
    result = runner.invoke(app, ["run", "mine.zip", "sam.zip", "-o", "sorted"])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_python_dash_m_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tripsort", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"tripsort {version('tripsort')}"
