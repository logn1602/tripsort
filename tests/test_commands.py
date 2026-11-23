"""The commands tripsort prints: spelled the way the user ran it, so they work as pasted."""

import ast
import re
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
from typer.testing import CliRunner

import tripsort
from tripsort import commands
from tripsort.cli import app
from tripsort.commands import command, command_prefix

PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")  # Typer forces colors on in CI
# A command written into a message by hand, rather than with tripsort.commands.command
SPELLED_OUT = re.compile(r"\btripsort (run|ingest|analyze|export|clean|people|models)\b")


@pytest.mark.parametrize(
    ("uv_set", "environment", "prefix"),
    [
        (True, ("uv", "cache", "archive-v0", "Ab3xQ9"), f"uvx --python {PYTHON} tripsort"),
        (False, ("uv", "cache", "archive-v0", "Ab3xQ9"), "tripsort"),  # uv did not start it
        (True, ("trip", ".venv"), "tripsort"),  # `uv run` in a project
        (True, ("uv", "tools", "tripsort"), "tripsort"),  # `uv tool install`: on PATH
        (False, ("venv",), "tripsort"),  # pip
    ],
)
def test_a_uvx_run_is_recognized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    uv_set: bool,
    environment: tuple[str, ...],
    prefix: str,
) -> None:
    if uv_set:
        monkeypatch.setenv("UV", str(tmp_path / "uv.exe"))
    else:
        monkeypatch.delenv("UV", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path.joinpath(*environment)))
    monkeypatch.setattr(commands, "has_video_extra", lambda: False)
    assert command_prefix() == prefix


def test_the_video_extra_is_found_by_pyav() -> None:
    assert commands.has_video_extra() is (find_spec("av") is not None)


def test_a_uvx_run_with_the_video_extra_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "run_by_uvx", lambda: True)
    monkeypatch.setattr(commands, "has_video_extra", lambda: True)
    assert command_prefix() == f'uvx --python {PYTHON} --from "tripsort[video]" tripsort'


def test_an_error_prints_the_video_extra_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI prints errors with Rich, which would read [video] as a style and drop it."""
    monkeypatch.setattr(commands, "run_by_uvx", lambda: True)
    monkeypatch.setattr(commands, "has_video_extra", lambda: True)
    result = CliRunner().invoke(app, ["people", "rename", str(tmp_path / "sorted"), "nobody", "x"])
    assert result.exit_code == 1
    expected = f'`uvx --python {PYTHON} --from "tripsort[video]" tripsort people list PROJECT`'
    assert expected in ANSI_ESCAPE.sub("", result.output)


def test_commands_are_ready_to_paste(uvx: str) -> None:
    assert command("export", "my trip/sorted") == f'{uvx} export "my trip/sorted"'
    assert command("people merge", "sorted", "person_07", "Mary Ann") == (
        f'{uvx} people merge sorted person_07 "Mary Ann"'
    )
    assert command("people list", "-sorted") == f"{uvx} people list -- -sorted"
    assert command("models download") == f"{uvx} models download"


def _docstrings(tree: ast.Module) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def test_no_message_spells_out_a_command() -> None:
    """Messages build their commands with tripsort.commands.command; docstrings may name them."""
    source = Path(tripsort.__file__).parent
    spelled = []
    for path in sorted(source.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstrings(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and SPELLED_OUT.search(node.value)
            ):
                spelled.append(f"{path.relative_to(source)}:{node.lineno}: {node.value!r}")
    assert spelled == []
