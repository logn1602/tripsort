"""Project-wide counts for summaries and the manifest, computed from the index."""

import sqlite3
from dataclasses import dataclass

from tripsort.dedupe.exact import exact_stats
from tripsort.ingest.zip_reader import READ_FAILURES


@dataclass(frozen=True)
class ProjectStats:
    sources: int
    files_found: int  # ingested zip entries (occurrences)
    unique: int  # distinct contents
    exact_duplicates: int
    exact_bytes_saved: int
    copy_groups: int
    near_copies: int  # recompressed copies left out of all/
    copy_bytes_saved: int
    kept: int  # files in all/
    images: int
    videos: int
    unreadable: int
    not_analyzed: int  # kept files without a pHash: videos, oversized images
    skipped: dict[str, int]  # reason -> number of zip entries
    bytes_in: int
    bytes_kept: int

    @property
    def bytes_saved(self) -> int:
        return self.bytes_in - self.bytes_kept

    @property
    def read_failures(self) -> int:
        return sum(n for reason, n in self.skipped.items() if reason in READ_FAILURES)


def project_stats(conn: sqlite3.Connection) -> ProjectStats:
    exact = exact_stats(conn)
    groups, copies, copy_bytes = conn.execute(
        "SELECT COUNT(DISTINCT gm.group_id), "
        "       COALESCE(SUM(CASE WHEN gm.is_keeper = 0 THEN 1 ELSE 0 END), 0), "
        "       COALESCE(SUM(CASE WHEN gm.is_keeper = 0 THEN m.bytes ELSE 0 END), 0) "
        "FROM group_members gm JOIN media_groups g ON g.id = gm.group_id "
        "JOIN media m ON m.id = gm.media_id WHERE g.kind = 'copy'"
    ).fetchone()
    kept, images, videos, not_analyzed, kept_bytes = conn.execute(
        "SELECT COUNT(*), "
        "       COALESCE(SUM(kind = 'image'), 0), COALESCE(SUM(kind = 'video'), 0), "
        "       COALESCE(SUM(phash IS NULL), 0), COALESCE(SUM(bytes), 0) "
        "FROM media WHERE status = 'ok' AND id IN (SELECT media_id FROM occurrences) "
        "AND id NOT IN (SELECT gm.media_id FROM group_members gm "
        "               JOIN media_groups g ON g.id = gm.group_id "
        "               WHERE g.kind = 'copy' AND gm.is_keeper = 0)"
    ).fetchone()
    unreadable, unreadable_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM media WHERE status = 'unreadable' "
        "AND id IN (SELECT media_id FROM occurrences)"
    ).fetchone()
    skipped = {
        str(reason): int(count)
        for reason, count in conn.execute(
            "SELECT reason, COUNT(*) FROM skipped_members GROUP BY reason ORDER BY reason"
        ).fetchall()
    }
    (sources,) = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
    return ProjectStats(
        sources=int(sources),
        files_found=exact.occurrences,
        unique=exact.unique,
        exact_duplicates=exact.duplicates,
        exact_bytes_saved=exact.bytes_saved,
        copy_groups=int(groups),
        near_copies=int(copies),
        copy_bytes_saved=int(copy_bytes),
        kept=int(kept),
        images=int(images),
        videos=int(videos),
        unreadable=int(unreadable),
        not_analyzed=int(not_analyzed),
        skipped=skipped,
        bytes_in=exact.bytes_in,
        bytes_kept=int(kept_bytes) + int(unreadable_bytes),
    )


def count(n: int, singular: str, plural: str | None = None) -> str:
    """'1 file', '2 files', '1,234 files'."""
    word = singular if n == 1 else (plural or f"{singular}s")
    return f"{n:,} {word}"


def human_bytes(n: int) -> str:
    """Decimal units, as file managers show them: 1.2 MB, 3.4 GB."""
    if abs(n) < 1000:
        return f"{n} B"
    value = n / 1000
    for unit in ("KB", "MB"):
        if abs(value) < 1000:
            return f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"
