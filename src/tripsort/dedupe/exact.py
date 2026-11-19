"""Exact duplicates: the same bytes in several places share one media row."""

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ExactStats:
    occurrences: int  # files found in the zips (ingested entries)
    unique: int  # distinct contents
    duplicates: int  # extra occurrences of content already seen
    bytes_in: int  # total size of every occurrence
    bytes_saved: int  # size of the extra occurrences


def exact_stats(conn: sqlite3.Connection) -> ExactStats:
    occurrences, bytes_in = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(m.bytes), 0) "
        "FROM occurrences o JOIN media m ON m.id = o.media_id"
    ).fetchone()
    unique, unique_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM media "
        "WHERE id IN (SELECT media_id FROM occurrences)"
    ).fetchone()
    return ExactStats(
        occurrences=int(occurrences),
        unique=int(unique),
        duplicates=int(occurrences) - int(unique),
        bytes_in=int(bytes_in),
        bytes_saved=int(bytes_in) - int(unique_bytes),
    )
