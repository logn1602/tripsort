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


def test_a_phase_2_index_is_upgraded_in_place(tmp_path: Path) -> None:
    """An index written by 0.1.0a1 (schema 1) keeps its rows and gains the people tables."""
    path = tmp_path / "index.db"
    old = sqlite3.connect(path, isolation_level=None)
    number, sql = db.migrations()[0]
    assert number == 1
    old.executescript(
        "BEGIN;\nCREATE TABLE schema_version (version INTEGER NOT NULL);\n"
        f"INSERT INTO schema_version (version) VALUES (0);\n{sql}\n"
        "UPDATE schema_version SET version = 1;\nCOMMIT;\n"
    )
    old.execute("INSERT INTO media (id, sha256, kind, bytes) VALUES (1, 'abc', 'image', 10)")
    old.execute(
        "INSERT INTO exported_files (relpath, fold_key, media_id, role, method, bytes, mtime_ns) "
        "VALUES ('all/a.jpg', 'all/a.jpg', 1, 'all', 'extract', 10, 5)"
    )
    old.close()

    conn = open_index(path)
    try:
        assert schema_version(conn) == latest_version() >= 2
        rows = conn.execute("SELECT relpath, role FROM exported_files").fetchall()
        assert [tuple(row) for row in rows] == [("all/a.jpg", "all")]
        conn.execute(
            "INSERT INTO exported_files (relpath, fold_key, media_id, role, method, bytes, "
            "mtime_ns) VALUES ('people/sam/a.jpg', 'people/sam/a.jpg', 1, 'person', "
            "'hardlink', 10, 5)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO exported_files (relpath, fold_key, media_id, role, method, "
                "bytes, mtime_ns) VALUES ('x', 'x', 1, 'bogus', 'copy', 1, 1)"
            )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(media)")}
        assert {"shot", "is_selfie", "is_front_camera"} <= columns
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert {"faces", "face_scans", "people", "ref_images", "exported_media"} <= tables
    finally:
        conn.close()


def schema(path: Path, version: int) -> sqlite3.Connection:
    """A fresh index at ``version``, migrated only that far."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    for number, sql in db.migrations()[:version]:
        bootstrap = (
            "CREATE TABLE schema_version (version INTEGER NOT NULL);\n"
            "INSERT INTO schema_version (version) VALUES (0);\n"
            if number == 1
            else ""
        )
        conn.executescript(
            f"BEGIN;\n{bootstrap}{sql}\nUPDATE schema_version SET version = {number};\nCOMMIT;\n"
        )
    return conn


def test_rebuilding_media_keeps_every_row_and_reference(tmp_path: Path) -> None:
    """Schema 5 rebuilds media (for the 'video' date source) while eight tables refer to it."""
    path = tmp_path / "index.db"
    old = schema(path, 4)
    old.execute(
        "INSERT INTO media (id, sha256, kind, bytes, taken_at, taken_at_source, shot, "
        "clip_selfie) VALUES (1, 'abc', 'image', 10, '2026-07-14T09:00:00', 'exif', 'solo', 0.25)"
    )
    old.execute("INSERT INTO media (id, sha256, kind, bytes) VALUES (2, 'def', 'video', 99)")
    old.execute(
        "INSERT INTO faces (media_id, model_id, face_index, x, y, w, h, score, landmarks, "
        "embedding) VALUES (1, 'm', 0, 0.1, 0.1, 0.2, 0.2, 0.9, '[]', x'00')"
    )
    old.execute("INSERT INTO media_groups (id, kind) VALUES (7, 'burst')")
    old.execute("INSERT INTO group_members (group_id, media_id, is_keeper) VALUES (7, 1, 1)")
    old.execute(
        "INSERT INTO categories (media_id, label, score, is_primary) VALUES (1, 'scenery', 0.9, 1)"
    )
    old.close()

    conn = open_index(path)
    try:
        assert schema_version(conn) == latest_version() >= 5
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        row = conn.execute(
            "SELECT sha256, taken_at, taken_at_source, shot, clip_selfie FROM media WHERE id = 1"
        ).fetchone()
        assert tuple(row) == ("abc", "2026-07-14T09:00:00", "exif", "solo", 0.25)
        assert conn.execute("SELECT COUNT(*) FROM faces WHERE media_id = 1").fetchone()[0] == 1
        assert conn.execute("SELECT is_keeper FROM group_members").fetchone()[0] == 1
        conn.execute(
            "UPDATE media SET video_created = '2026-07-14T10:00:00', taken_at_source = 'video' "
            "WHERE id = 2"
        )
        with pytest.raises(sqlite3.IntegrityError):  # references still hold
            conn.execute(
                "INSERT INTO faces (media_id, model_id, face_index, x, y, w, h, score, "
                "landmarks, embedding) VALUES (9, 'm', 0, 0, 0, 1, 1, 1, '[]', x'00')"
            )
        with pytest.raises(sqlite3.IntegrityError):  # and so do the CHECK constraints
            conn.execute("UPDATE media SET taken_at_source = 'guess' WHERE id = 2")
        conn.execute("DELETE FROM media_groups WHERE id = 7")  # cascades as before
        assert conn.execute("SELECT COUNT(*) FROM group_members").fetchone()[0] == 0
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert {"quality", "video_scans", "live_photos", "media_taken"} <= tables
    finally:
        conn.close()
