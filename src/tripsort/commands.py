"""The tripsort commands that messages ask the user to run, spelled so they work as pasted.

The README's quick start runs tripsort through uvx, which puts nothing on PATH, so a bare
``tripsort people merge ...`` fails after it. A uvx run is recognized by two signs: Python
runs from a throwaway environment in uv's cache (a folder under ``archive-v0``), and uv
sets ``UV`` for the process it starts. ``shutil.which("tripsort")`` cannot tell, because
uvx puts that environment's scripts folder on PATH for the process it starts. pip,
``uv tool install``, and project environments get plain ``tripsort``, as does anything
these signs do not recognize.

Standard library only: the error messages of light modules use it too.
"""

import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path

_PLAIN_ARG = re.compile(r"^[\w./\\:-]+$")  # safe unquoted in POSIX shells, PowerShell, and cmd
_UV_CACHE_BUCKET = re.compile(r"^archive-v\d+$")


def run_by_uvx() -> bool:
    """Whether uvx started this process: ``UV`` is set and Python runs from uv's cache."""
    return bool(os.environ.get("UV")) and any(
        _UV_CACHE_BUCKET.match(part) for part in Path(sys.prefix).parts
    )


def has_video_extra() -> bool:
    """Whether PyAV, which the ``video`` extra brings, is installed."""
    try:
        return find_spec("av") is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken or half-imported av
        return False


def command_prefix() -> str:
    """``tripsort``, or after a uvx run ``uvx --python 3.13 tripsort``: the running Python's
    version, plus ``--from "tripsort[video]"`` when the video extra is installed."""
    if not run_by_uvx():
        return "tripsort"
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    extra = f" --from {shell_arg('tripsort[video]')}" if has_video_extra() else ""
    return f"uvx --python {python}{extra} tripsort"


def shell_arg(text: str) -> str:
    """``text`` as one command-line argument to paste into a shell: as is when it is plain,
    else in double quotes (single quotes when it holds a double quote)."""
    if _PLAIN_ARG.match(text):
        return text
    if '"' not in text:
        return f'"{text}"'
    return "'" + text.replace("'", "'\"'\"'") + "'"


def command(subcommand: str, *args: str) -> str:
    """``tripsort SUBCOMMAND ARGS...`` as the user can paste it: the prefix above, the
    subcommand as given (``"people merge"``), and each argument quoted for a shell, after
    ``--`` when one starts with a dash, so it is not read as an option."""
    end = ["--"] if any(a.startswith("-") for a in args) else []
    return " ".join([command_prefix(), subcommand, *end, *(shell_arg(a) for a in args)])
