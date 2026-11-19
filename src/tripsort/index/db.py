"""Open the project index, apply migrations, and run transactions.

Connections run in autocommit mode (``isolation_level=None``); every write goes through
``transaction()``, so the behavior is the same on Python 3.11 to 3.13.
"""

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

from tripsort.errors import IndexVersionError

INDEX_DIR_NAME = ".tripsort"
INDEX_FILE_NAME = "index.db"

_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def index_path(project_dir: Path) -> Path:
    return project_dir / INDEX_DIR_NAME / INDEX_FILE_NAME


def migrations() -> list[tuple[int, str]]:
    """The packaged migrations as ``(number, sql)``, in order."""
    found: list[tuple[int, str]] = []
    for entry in files("tripsort.index").joinpath("migrations").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))
    found.sort()
    numbers = [number for number, _ in found]
    if numbers != list(range(1, len(numbers) + 1)):  # pragma: no cover - guards packaging
        raise RuntimeError(f"migrations must be numbered 1..n without gaps, found {numbers}")
    return found


def latest_version() -> int:
    return migrations()[-1][0]


def open_index(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the index at ``path`` and bring its schema up to date."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        if is_new:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if exists is None:
        return 0
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return int(row[0]) if row is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every migration newer than the index, each in its own transaction."""
    current = schema_version(conn)
    available = migrations()
    latest = available[-1][0]
    if current > latest:
        raise IndexVersionError(
            f"this index uses schema version {current}, but this tripsort only knows up to "
            f"{latest}; upgrade tripsort to open it"
        )
    for number, sql in available:
        if number <= current:
            continue
        bootstrap = (
            "CREATE TABLE schema_version (version INTEGER NOT NULL);\n"
            "INSERT INTO schema_version (version) VALUES (0);\n"
            if number == 1
            else ""
        )
        script = (
            f"BEGIN;\n{bootstrap}{sql}\nUPDATE schema_version SET version = {number};\nCOMMIT;\n"
        )
        try:
            conn.executescript(script)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return latest


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN`` ... ``COMMIT``, rolled back if the block raises (including Ctrl+C)."""
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:  # SQLite rolls back by itself after some errors
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row is not None else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
