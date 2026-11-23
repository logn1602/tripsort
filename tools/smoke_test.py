# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""End-to-end check of an installed tripsort: build a tiny zip, run it, check the output.

Uses only the standard library (the PNGs are written with zlib), so it runs where tripsort
was installed from a wheel and nothing else. It catches packaging mistakes, such as a
migration or config file missing from the wheel, that the test suite cannot see. It also
checks every photo was scanned for faces, so the face models must download (or already be
in the model cache) and pass their SHA-256 check.

    uv run --with dist/tripsort-0.3.0-py3-none-any.whl tools/smoke_test.py   # a built wheel
    uv run python tools/smoke_test.py                                     # the dev install
    uv run tools/smoke_test.py uvx --from tripsort==0.3.0 tripsort       # any command

With no command, it runs ``python -m tripsort`` with this interpreter.
"""

import json
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path


def png(width: int, height: int, pixel: Callable[[int, int], tuple[int, int, int]]) -> bytes:
    rows = b"".join(
        b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def make_trip_zip(path: Path) -> None:
    """Three distinct images (different aspect ratios), one exact duplicate, two skips."""
    a = png(64, 48, lambda x, y: (x * 4, 128, 255 - x * 4))
    b = png(48, 64, lambda x, y: (y * 4, 255 - y * 4, 64))
    c = png(64, 64, lambda x, y: (255, 255, 255) if (x // 8 + y // 8) % 2 else (0, 0, 0))
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("DCIM/a.png", a)
        zf.writestr("DCIM/b.png", b)
        zf.writestr("DCIM/c.png", c)
        zf.writestr("Backup/a-again.png", a)
        zf.writestr("__MACOSX/DCIM/._a.png", b"junk")
        zf.writestr("notes.txt", b"not a photo")


def run_smoke(tripsort: list[str], workdir: Path, env: dict[str, str] | None = None) -> None:
    """Run the checks, raising AssertionError (with the command's output) on any failure."""

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*tripsort, *args],
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            env=env,
        )

    version = run("--version")
    assert version.returncode == 0, version.stderr
    assert version.stdout.startswith("tripsort "), version.stdout

    zip_path = workdir / "trip.zip"
    out = workdir / "sorted"
    make_trip_zip(zip_path)
    first = run("run", str(zip_path), "-o", str(out))
    assert first.returncode == 0, f"exit {first.returncode}\n{first.stdout}\n{first.stderr}"
    kept = sorted(p.name for p in (out / "all").iterdir())
    assert len(kept) == 3, kept
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]
    assert (summary["unique"], summary["exact_duplicates"]) == (3, 1), summary
    assert summary["skipped"] == {"junk": 1, "unsupported_type": 1}, summary
    assert summary["faces_not_scanned"] == 0, summary  # the face models loaded
    assert summary["shots"]["none"] == 3, summary  # no faces in these PNGs

    second = run("run", str(zip_path), "-o", str(out))
    assert second.returncode == 0, second.stderr
    assert "already indexed" in second.stdout, second.stdout
    assert "Wrote 0 files" in second.stdout, second.stdout


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    tripsort = args or [sys.executable, "-m", "tripsort"]
    with tempfile.TemporaryDirectory(prefix="tripsort-smoke-") as tmp:
        try:
            run_smoke(tripsort, Path(tmp))
        except AssertionError as exc:
            print(f"smoke test FAILED: {exc}", file=sys.stderr)
            return 1
    print("smoke test passed: tripsort run works end to end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
