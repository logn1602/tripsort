import sqlite3
from pathlib import Path

import pytest

from tripsort.errors import IndexVersionError
from tripsort.index import db
from tripsort.index.db import (
    get_meta,
    latest_version,
    migrate,
    open_index,
    schema_version,
    set_meta,
    transaction,
)


def test_new_index_is_migrated_and_uses_wal(tmp_path: Path) -> None:
    path = tmp_path / ".tripsort" / "index.db"
    conn = open_index(path)
    try:
        assert path.is_file()
        assert schema_version(conn) == latest_version() >= 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert {"sources", "media", "occurrences", "skipped_members", "exported_files"} <= tables
    finally:
        conn.close()


def test_reopening_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    open_index(path).close()
    conn = open_index(path)
    try:
        assert schema_version(conn) == latest_version()
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
    finally:
        conn.close()


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    conn = open_index(tmp_path / "index.db")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO occurrences (source_id, member_index, media_id, member_path) "
                "VALUES (1, 0, 1, 'x.jpg')"
            )
    finally:
        conn.close()


def test_index_from_a_newer_tripsort_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    conn = open_index(path)
    conn.execute("UPDATE schema_version SET version = ?", (latest_version() + 1,))
    conn.close()
    with pytest.raises(IndexVersionError, match="upgrade tripsort"):
        open_index(path)


def test_a_failing_migration_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = open_index(tmp_path / "index.db")
    real = db.migrations()
    broken = (len(real) + 1, "CREATE TABLE half_done (x INTEGER);\nTHIS IS NOT SQL;")
    monkeypatch.setattr(db, "migrations", lambda: [*real, broken])
    try:
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn)
        assert schema_version(conn) == len(real)
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name = 'half_done'").fetchone()
            is None
        )
        assert not conn.in_transaction
    finally:
        conn.close()


def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    conn = open_index(tmp_path / "index.db")
    try:

        def fail_inside_transaction() -> None:
            with transaction(conn):
                set_meta(conn, "k", "v")
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            fail_inside_transaction()
        assert get_meta(conn, "k") is None
        with transaction(conn):
            set_meta(conn, "k", "v1")
            set_meta(conn, "k", "v2")
        assert get_meta(conn, "k") == "v2"
    finally:
        conn.close()


def test_packaged_migrations_are_numbered_from_one() -> None:
    numbers = [number for number, _ in db.migrations()]
    assert numbers == list(range(1, len(numbers) + 1))
