# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check that a tripsort release on TestPyPI installs and runs.

Only tripsort itself comes from TestPyPI: its wheel is downloaded with --no-deps, then
installed from that local file into a throwaway environment where every dependency is
resolved from PyPI. TestPyPI can lag for a few minutes after an upload, so the download
is retried until --timeout runs out.

Usage, from the repo root on any OS:

    uv run tools/verify_testpypi.py              # version from pyproject.toml
    uv run tools/verify_testpypi.py 0.0.1 --python 3.13
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

TESTPYPI_SIMPLE = "https://test.pypi.org/simple/"
PYPI_SIMPLE = "https://pypi.org/simple/"
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Index settings in the environment could add other indexes. They are dropped, and pip is
# pointed at no config file, so the only indexes in play are the ones named on the command
# lines below. (`uv --no-config` already skips uv.toml and pyproject.toml settings.)
UV_INDEX_ENV = (
    "UV_INDEX",
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX_STRATEGY",
    "UV_NO_INDEX",
    "UV_CONFIG_FILE",
)


def isolated_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in UV_INDEX_ENV and not key.upper().startswith("PIP_")
    }
    env["PIP_CONFIG_FILE"] = os.devnull  # pip then loads no config files at all
    return env


def project_version() -> str:
    with PYPROJECT.open("rb") as f:
        return str(tomllib.load(f)["project"]["version"])


def uv_tool_run(python: str) -> list[str]:
    """`uv tool run` in a fresh environment that ignores local uv config and resolves from PyPI."""
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv was not found on PATH")
    return [
        uv,
        "tool",
        "run",
        "--isolated",
        "--no-config",
        "--python",
        python,
        "--default-index",
        PYPI_SIMPLE,
    ]


def download_wheel(version: str, dest: Path, python: str, timeout: float, interval: float) -> Path:
    """Download only the tripsort wheel from TestPyPI, retrying while the index catches up."""
    cmd = [
        *uv_tool_run(python),
        "pip",
        "download",
        f"tripsort=={version}",
        "--no-deps",
        "--only-binary=:all:",
        "--no-cache-dir",  # a cached "not found" index page would defeat the retries
        "--disable-pip-version-check",
        "--index-url",
        TESTPYPI_SIMPLE,
        "--dest",
        str(dest),
    ]
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        attempt += 1
        print(f"[attempt {attempt}] downloading tripsort {version} from TestPyPI", flush=True)
        if subprocess.run(cmd, check=False, env=isolated_env()).returncode == 0:
            wheels = sorted(dest.glob("tripsort-*.whl"))
            if len(wheels) == 1:
                return wheels[0]
        if time.monotonic() + interval > deadline:
            sys.exit(f"tripsort {version} was not downloadable from TestPyPI after {attempt} tries")
        print(f"not available yet; retrying in {interval:.0f}s", flush=True)
        time.sleep(interval)


def run_from_wheel(wheel: Path, python: str) -> str:
    """Install the wheel with its dependencies from PyPI and return `tripsort --version`."""
    cmd = [*uv_tool_run(python), "--from", str(wheel), "tripsort", "--version"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=isolated_env())
    if result.returncode != 0:
        sys.exit(f"tripsort --version failed with exit code {result.returncode}:\n{result.stderr}")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that a tripsort release on TestPyPI installs and runs."
    )
    parser.add_argument(
        "version", nargs="?", help="version to check (default: the one in pyproject.toml)"
    )
    parser.add_argument(
        "--python", default="3.11", help="Python for the test install (default: 3.11)"
    )
    parser.add_argument(
        "--timeout", type=float, default=300, help="seconds to keep retrying (default: 300)"
    )
    parser.add_argument(
        "--interval", type=float, default=15, help="seconds between tries (default: 15)"
    )
    args = parser.parse_args()

    version = args.version or project_version()
    with tempfile.TemporaryDirectory() as tmp:
        wheel = download_wheel(version, Path(tmp), args.python, args.timeout, args.interval)
        print(f"downloaded {wheel.name}; installing it with dependencies from PyPI", flush=True)
        output = run_from_wheel(wheel, args.python)

    expected = f"tripsort {version}"
    if output != expected:
        sys.exit(f"expected {expected!r} but got {output!r}")
    print(f"OK: {output} (tripsort from TestPyPI, dependencies from PyPI, Python {args.python})")


if __name__ == "__main__":
    main()
