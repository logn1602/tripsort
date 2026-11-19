"""Ingest zips into the index: hash every entry, decode only content not seen before.

* A zip whose fingerprint is already fully ingested is skipped without reading any entry.
* An interrupted zip resumes: entries that already have a row are not read again.
* Results are committed every ``batch_size`` entries, so Ctrl+C loses one batch at most.
"""

import sqlite3
import time
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from tripsort.ingest.decode import UnreadableImageError, analyze_image
from tripsort.ingest.hashing import HashedBytes, hash_chunks
from tripsort.ingest.zip_reader import (
    READ_FAILURES,
    SKIP_CORRUPT,
    SKIP_UNSUPPORTED_COMPRESSION,
    Member,
    ZipSource,
    skip_hint,
)
from tripsort.progress import NullProgress, Progress

BATCH_SIZE = 50
MAX_BUFFERED_IMAGE_BYTES = 512 * 1024 * 1024
VIDEO_NOTE = "video: checked for exact duplicates only until video support arrives"
_VIDEO_FORMATS = {".mp4": "MP4", ".m4v": "MP4", ".mov": "MOV"}


@dataclass(frozen=True)
class Problem:
    """A zip entry that could not be read or decoded, and what to do about it."""

    source: str
    member: str
    message: str


@dataclass
class SourceReport:
    label: str
    zip_name: str
    status: str  # 'new', 'resumed', or 'already indexed'
    members: int
    new_files: int = 0  # content not seen before (decoded, or passed through)
    known_files: int = 0  # content already in the index: only an occurrence was added
    skipped: int = 0
    unreadable: int = 0
    bytes_read: int = 0


@dataclass
class IngestResult:
    sources: list[SourceReport]
    decoded: int  # images whose pixels were decoded in this run
    problems: list[Problem]
    seconds: float
    bytes_read: int
    peak_rss_bytes: int | None = None

    @property
    def new_files(self) -> int:
        return sum(report.new_files for report in self.sources)

    @property
    def known_files(self) -> int:
        return sum(report.known_files for report in self.sources)

    @property
    def files_read(self) -> int:
        return self.new_files + self.known_files

    def summary(self) -> str:
        lines = []
        for report in self.sources:
            if report.status == "already indexed":
                lines.append(f"{report.zip_name}: already indexed, nothing to read")
                continue
            parts = [f"{report.new_files:,} new", f"{report.known_files:,} already indexed"]
            if report.skipped:
                parts.append(f"{report.skipped:,} skipped")
            if report.unreadable:
                parts.append(f"{report.unreadable:,} unreadable")
            verb = "resumed" if report.status == "resumed" else "read"
            lines.append(f"{report.zip_name} ({report.label}): {verb}, {', '.join(parts)}")
        return "\n".join(lines)


@dataclass
class _Run:
    conn: sqlite3.Connection
    progress: Progress
    batch_size: int
    decoded: int = 0
    problems: list[Problem] = field(default_factory=list)


def ingest_zips(
    conn: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    progress: Progress | None = None,
    batch_size: int = BATCH_SIZE,
) -> IngestResult:
    """Ingest ``paths`` into the index behind ``conn``. Raises ``ZipError`` before any work
    if a path is missing or is not a zip."""
    started = time.perf_counter()
    run = _Run(conn, progress or NullProgress(), batch_size)
    opened: list[ZipSource] = []
    try:
        for path in paths:  # open every zip first, so a bad path fails before any work
            opened.append(ZipSource(Path(path).resolve()))
        reports: list[SourceReport] = []
        work: list[tuple[ZipSource, int, SourceReport, list[Member]]] = []
        seen: set[str] = set()
        for source in opened:
            if source.fingerprint in seen:
                continue
            seen.add(source.fingerprint)
            source_id, label, status = _register(conn, source)
            report = SourceReport(
                label=label,
                zip_name=source.path.name,
                status="already indexed" if status == "complete" else status,
                members=len(source.members),
            )
            reports.append(report)
            if status == "complete":
                continue
            done = _done_indexes(conn, source_id)
            pending = [member for member in source.members if member.index not in done]
            pending.sort(key=lambda member: member.info.header_offset)
            work.append((source, source_id, report, pending))

        run.progress.start("ingest", sum(m.size for *_, pending in work for m in pending), "bytes")
        for source, source_id, report, pending in work:
            _ingest_source(run, source, source_id, report, pending)
        run.progress.finish()
    finally:
        for source in opened:
            source.close()
    return IngestResult(
        sources=reports,
        decoded=run.decoded,
        problems=run.problems,
        seconds=time.perf_counter() - started,
        bytes_read=sum(report.bytes_read for report in reports),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _register(conn: sqlite3.Connection, source: ZipSource) -> tuple[int, str, str]:
    """Find or create the source row. Returns (id, label, 'new' | 'resumed' | 'complete')."""
    row = conn.execute(
        "SELECT id, label, status, zip_path FROM sources WHERE fingerprint = ?",
        (source.fingerprint,),
    ).fetchone()
    if row is not None:
        source_id, label, status, zip_path = int(row[0]), str(row[1]), str(row[2]), row[3]
        if zip_path != str(source.path):  # the zip moved or was renamed
            conn.execute(
                "UPDATE sources SET zip_path = ?, zip_name = ? WHERE id = ?",
                (str(source.path), source.path.name, source_id),
            )
        return source_id, label, "complete" if status == "complete" else "resumed"
    label = _unique_label(conn, source.path.stem)
    cursor = conn.execute(
        "INSERT INTO sources (fingerprint, zip_path, zip_name, label, bytes, members, added_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            source.fingerprint,
            str(source.path),
            source.path.name,
            label,
            source.bytes,
            len(source.members),
            _now(),
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid, label, "new"


def _unique_label(conn: sqlite3.Connection, stem: str) -> str:
    base = stem.strip() or "zip"
    label, n = base, 1
    while conn.execute("SELECT 1 FROM sources WHERE label = ?", (label,)).fetchone():
        n += 1
        label = f"{base}-{n}"
    return label


def _done_indexes(conn: sqlite3.Connection, source_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT member_index FROM occurrences WHERE source_id = ? "
        "UNION SELECT member_index FROM skipped_members WHERE source_id = ?",
        (source_id, source_id),
    ).fetchall()
    return {int(row[0]) for row in rows}


def _ingest_source(
    run: _Run, source: ZipSource, source_id: int, report: SourceReport, pending: list[Member]
) -> None:
    conn = run.conn
    conn.execute("BEGIN")
    try:
        for count, member in enumerate(pending, start=1):
            _ingest_member(run, source, source_id, report, member)
            run.progress.advance(member.size)
            if count % run.batch_size == 0:
                conn.execute("COMMIT")
                conn.execute("BEGIN")
        conn.execute(
            "UPDATE sources SET status = 'complete', completed_at = ? WHERE id = ?",
            (_now(), source_id),
        )
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _ingest_member(
    run: _Run, source: ZipSource, source_id: int, report: SourceReport, member: Member
) -> None:
    if member.skip is not None:
        _skip(run, source_id, report, member, member.skip, member.detail)
        return
    keep = member.kind == "image" and member.size <= MAX_BUFFERED_IMAGE_BYTES
    try:
        hashed = hash_chunks(source.iter_chunks(member), keep=keep)
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        _skip(run, source_id, report, member, SKIP_CORRUPT, _describe(exc))
        return
    except (NotImplementedError, RuntimeError) as exc:  # e.g. a missing bz2 or lzma module
        _skip(run, source_id, report, member, SKIP_UNSUPPORTED_COMPRESSION, _describe(exc))
        return
    report.bytes_read += hashed.size

    row = run.conn.execute("SELECT id FROM media WHERE sha256 = ?", (hashed.sha256,)).fetchone()
    if row is not None:
        media_id = int(row[0])
        report.known_files += 1
    else:
        media_id = _insert_media(run, report, member, hashed)
        report.new_files += 1
    run.conn.execute(
        "INSERT INTO occurrences "
        "(source_id, member_index, media_id, member_path, member_raw, zip_mtime) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, member.index, media_id, member.path, member.raw_name, member.mtime),
    )


def _insert_media(run: _Run, report: SourceReport, member: Member, hashed: HashedBytes) -> int:
    values: dict[str, object] = {"sha256": hashed.sha256, "bytes": hashed.size}
    if member.kind == "video":
        suffix = PurePosixPath(member.path).suffix.lower()
        values.update(kind="video", format=_VIDEO_FORMATS.get(suffix), note=VIDEO_NOTE)
    elif hashed.data is None:
        values.update(kind="image", note="too large to decode; checked for exact duplicates only")
    else:
        values["kind"] = "image"
        try:
            analysis = analyze_image(hashed.data)
        except UnreadableImageError as exc:
            values.update(status="unreadable", note=str(exc))
            report.unreadable += 1
            run.problems.append(
                Problem(
                    report.label,
                    member.path,
                    f"could not decode it ({exc}); export copies it to unreadable/ for review",
                )
            )
        else:
            exif = analysis.exif
            values.update(
                format=analysis.format,
                width=analysis.width,
                height=analysis.height,
                exif_datetime=exif.datetime,
                exif_offset=exif.offset,
                has_exif=int(exif.has_camera_exif),
                camera_make=exif.make,
                camera_model=exif.model,
                lens_model=exif.lens,
                exposure_time=exif.exposure_time,
                iso=exif.iso,
                gps_lat=exif.gps_lat,
                gps_lon=exif.gps_lon,
                phash=analysis.phash,
                note=analysis.note,
            )
            if analysis.decoded:
                run.decoded += 1
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cursor = run.conn.execute(
        f"INSERT INTO media ({columns}) VALUES ({placeholders})", tuple(values.values())
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _skip(
    run: _Run,
    source_id: int,
    report: SourceReport,
    member: Member,
    reason: str,
    detail: str | None,
) -> None:
    run.conn.execute(
        "INSERT INTO skipped_members "
        "(source_id, member_index, member_path, member_raw, bytes, reason, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_id, member.index, member.path, member.raw_name, member.size, reason, detail),
    )
    report.skipped += 1
    if reason in READ_FAILURES:
        hint = skip_hint(reason, detail)
        message = f"not read ({reason.replace('_', ' ')})" + (f". {hint}" if hint else "")
        run.problems.append(Problem(report.label, member.path, message))


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
