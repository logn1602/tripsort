"""Ingest zips into the index: hash every entry, decode only content not seen before.

* A zip whose fingerprint is already fully ingested is skipped without reading any entry.
* An interrupted zip resumes: entries that already have a row are not read again.
* Results are committed every ``batch_size`` entries, so Ctrl+C loses one batch at most.
* With a face analyzer, each new photo is scanned for faces in the same decode. Photos
  indexed without one (an older tripsort, no models at the time, or a new face model) are
  scanned afterwards, read back from the zips the index already knows.
* The same goes for the CLIP image embedding: new photos are embedded in the same decode
  (in batches, each committed with its photos), and photos without one are embedded
  afterwards, in the same pass that catches up on faces, so each is read back only once.
"""

import sqlite3
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from tripsort.clip.store import ClipBatch, pending_clip_scans
from tripsort.clip.types import CLIP_MODEL_ID, ClipPixels, ImageEncoder
from tripsort.commands import command
from tripsort.config import VideoSettings
from tripsort.cull.quality import QUALITY_VERSION, Quality, store_quality
from tripsort.faces.store import pending_scans, store_scan
from tripsort.faces.types import FACE_MODEL_ID, FaceAnalyzer, FaceScan
from tripsort.ingest.decode import UnreadableImageError, analyze_image
from tripsort.ingest.hashing import HashedBytes, hash_chunks
from tripsort.ingest.sources import ZipCache
from tripsort.ingest.zip_reader import (
    READ_FAILURES,
    SKIP_CORRUPT,
    SKIP_UNSUPPORTED_COMPRESSION,
    Member,
    ZipSource,
    skip_hint,
)
from tripsort.progress import NullProgress, Progress
from tripsort.stats import count
from tripsort.video import sample as video_sample
from tripsort.video.scan import VideoScanStats, pending_videos, scan_videos, video_scan_id

BATCH_SIZE = 50
MAX_BUFFERED_IMAGE_BYTES = 512 * 1024 * 1024
VIDEO_NOTE = "video: checked for exact duplicates only; the [video] extra (PyAV) samples it"
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
    faces_scanned: int = 0  # photos scanned for faces in this run
    faces_found: int = 0
    faces_caught_up: int = 0  # of those, photos from earlier ingests
    faces_pending: int = 0  # photos still waiting for a face scan after this run
    face_problem: str | None = None  # why no faces were scanned at all
    clip_embedded: int = 0  # photos given a CLIP embedding in this run
    clip_caught_up: int = 0  # of those, photos from earlier ingests
    clip_pending: int = 0  # photos still waiting for a CLIP embedding after this run
    clip_problem: str | None = None  # why no CLIP embeddings were made at all
    videos: VideoScanStats = field(default_factory=VideoScanStats)
    videos_waiting: int = 0  # videos not sampled yet (no PyAV, or no models)
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
        v = self.videos
        if v.sampled:
            lines.append(
                f"Sampled {count(v.sampled, 'video')} ({count(v.frames, 'frame')}) for faces "
                "and CLIP."
            )
        if v.failed:
            lines.append(f"{count(v.failed, 'video')} could not be read (see manifest.json).")
        if self.videos_waiting and not video_sample.available():
            lines.append(
                f"{count(self.videos_waiting, 'video')} checked for exact duplicates only; "
                "install the video extra (pip install 'tripsort[video]') to find people and "
                "categories in them."
            )
        if self.face_problem:
            lines.append(f"Faces were not scanned: {self.face_problem}")
        if self.faces_caught_up:
            lines.append(
                f"Scanned {count(self.faces_caught_up, 'photo')} from earlier zips for faces."
            )
        if self.faces_pending:
            lines.append(
                f"{count(self.faces_pending, 'photo')} still "
                f"{'waits' if self.faces_pending == 1 else 'wait'} for a face scan, so people/ "
                f"and shots/ leave them out until a later `{command('ingest')}` scans them."
            )
        if self.clip_problem:
            lines.append(f"Photos were not embedded for categories and bursts: {self.clip_problem}")
        if self.clip_caught_up:
            lines.append(
                f"Embedded {count(self.clip_caught_up, 'photo')} from earlier zips for "
                "categories and bursts."
            )
        if self.clip_pending:
            lines.append(
                f"{count(self.clip_pending, 'photo')} still "
                f"{'waits' if self.clip_pending == 1 else 'wait'} for a CLIP embedding, so "
                f"categories/ and bursts leave them out until a later `{command('ingest')}`."
            )
        return "\n".join(lines)


@dataclass
class _Run:
    conn: sqlite3.Connection
    progress: Progress
    batch_size: int
    faces: FaceAnalyzer | None = None
    clip: ClipBatch | None = None
    decoded: int = 0
    faces_scanned: int = 0
    faces_found: int = 0
    problems: list[Problem] = field(default_factory=list)

    def commit(self) -> None:
        """Commit the open transaction, with the CLIP embeddings of its photos."""
        if self.clip is not None:
            self.clip.flush()
        self.conn.execute("COMMIT")


def ingest_zips(
    conn: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    progress: Progress | None = None,
    batch_size: int = BATCH_SIZE,
    faces: Callable[[], FaceAnalyzer | None] | None = None,
    clip: Callable[[], ImageEncoder | None] | None = None,
    video: VideoSettings | None = None,
) -> IngestResult:
    """Ingest ``paths`` into the index behind ``conn``, then scan for faces, and embed with
    CLIP, the photos that have neither yet. Raises ``ZipError`` before any work if a path is
    missing or is not a zip.

    ``faces`` loads the face analyzer and ``clip`` the CLIP image encoder (each returns None
    when it cannot). Each is called once, before the first photo is read, and only when
    there is something for it to do. With ``video`` and PyAV installed, videos are probed
    and sampled after the photos.
    """
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

        has_new = any(pending for *_, pending in work)
        videos_due = (
            video is not None
            and video_sample.available()
            and bool(pending_videos(conn, video_scan_id(FACE_MODEL_ID, CLIP_MODEL_ID, video)))
        )
        if faces is not None and (has_new or videos_due or pending_scans(conn, FACE_MODEL_ID)):
            run.faces = faces()  # may download the models, with a progress bar of its own
        if clip is not None and (has_new or videos_due or pending_clip_scans(conn, CLIP_MODEL_ID)):
            encoder = clip()  # the same
            run.clip = ClipBatch(conn, encoder) if encoder is not None else None
        run.progress.start("ingest", sum(m.size for *_, pending in work for m in pending), "bytes")
        for source, source_id, report, pending in work:
            _ingest_source(run, source, source_id, report, pending)
        run.progress.finish()
    finally:
        for source in opened:
            source.close()
    scanned_inline = run.faces_scanned
    embedded_inline = run.clip.embedded if run.clip is not None else 0
    _scan_pending(run)
    embedded = run.clip.embedded if run.clip is not None else 0
    videos = VideoScanStats()
    if video is not None and video_sample.available():
        encoder = run.clip.encoder if run.clip is not None else None
        videos = scan_videos(conn, video, run.faces, encoder, run.progress)
    (waiting,) = conn.execute(
        "SELECT COUNT(*) FROM media m WHERE m.kind = 'video' AND m.status = 'ok' "
        "AND m.id IN (SELECT media_id FROM occurrences) AND NOT EXISTS ("
        "  SELECT 1 FROM video_scans s WHERE s.media_id = m.id "
        "  AND s.status IN ('ok', 'live_photo', 'not_decoded'))"
    ).fetchone()
    return IngestResult(
        sources=reports,
        decoded=run.decoded,
        problems=run.problems,
        seconds=time.perf_counter() - started,
        bytes_read=sum(report.bytes_read for report in reports),
        faces_scanned=run.faces_scanned,
        faces_found=run.faces_found,
        faces_caught_up=run.faces_scanned - scanned_inline,
        clip_embedded=embedded,
        clip_caught_up=embedded - embedded_inline,
        videos=videos,
        videos_waiting=int(waiting),
    )


def _scan_pending(run: _Run) -> None:
    """Scan for faces, embed with CLIP, and score the sharpness and exposure of the photos
    that lack a face scan or CLIP scan for the loaded models, or scores of this version,
    zip by zip in file order. A photo that needs several is read and decoded once."""
    conn = run.conn
    face_model = run.faces.model_id if run.faces is not None else ""
    clip_model = run.clip.encoder.model_id if run.clip is not None else ""
    rows = conn.execute(
        "SELECT m.id, m.sha256, "
        "  ? != '' AND NOT EXISTS (SELECT 1 FROM face_scans s WHERE s.media_id = m.id "
        "                          AND s.model_id = ?), "
        "  ? != '' AND NOT EXISTS (SELECT 1 FROM clip_scans c WHERE c.media_id = m.id "
        "                          AND c.model_id = ?), "
        "  m.bytes <= ? AND NOT EXISTS (SELECT 1 FROM quality q WHERE q.media_id = m.id "
        "                               AND q.version = ?) "
        "FROM media m WHERE m.kind = 'image' "
        "AND m.status = 'ok' AND m.id IN (SELECT media_id FROM occurrences) ORDER BY m.id",
        (
            face_model,
            face_model,
            clip_model,
            clip_model,
            MAX_BUFFERED_IMAGE_BYTES,
            QUALITY_VERSION,
        ),
    ).fetchall()
    rows = [row for row in rows if row[2] or row[3] or row[4]]
    if not rows:
        return
    with ZipCache(conn) as zips:
        located = []
        unreachable: Counter[tuple[str, ...]] = Counter()
        for media_id, sha256, need_faces, need_clip, need_quality in rows:
            found = zips.find(int(media_id))
            if found is None:
                unreachable[tuple(zips.missing_names(int(media_id)))] += 1
                continue
            source, member = found
            located.append(
                (
                    str(source.path),
                    member.info.header_offset,
                    int(media_id),
                    str(sha256),
                    member,
                    bool(need_faces),
                    bool(need_clip),
                    bool(need_quality),
                )
            )
        located.sort(key=lambda item: item[:3])
        run.progress.start("scan", len(located), "files")
        conn.execute("BEGIN")
        try:
            for done, item in enumerate(located, start=1):
                zip_path, _, media_id, sha256, member, need_faces, need_clip, need_quality = item
                scan, pixels, quality, read = _rescan(
                    zips,
                    media_id,
                    sha256,
                    member.size,
                    run.faces if need_faces else None,
                    need_clip,
                )
                if not read:
                    run.problems.append(
                        Problem(
                            Path(zip_path).name,
                            member.path,
                            "could not be read back from the zip, so it was not scanned",
                        )
                    )
                if scan is not None and run.faces is not None:
                    store_scan(conn, media_id, run.faces.model_id, scan)
                    run.faces_scanned += 1
                    run.faces_found += len(scan.faces)
                if pixels is not None and run.clip is not None:
                    run.clip.add(media_id, pixels)
                if quality is not None and need_quality:
                    store_quality(conn, media_id, quality)
                run.progress.advance(1)
                if done % run.batch_size == 0:
                    run.commit()
                    conn.execute("BEGIN")
            run.commit()
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            run.progress.finish()
    for names, photos in sorted(unreachable.items()):
        run.problems.append(
            Problem(
                ", ".join(names),
                count(photos, "photo"),
                "not scanned: the zip is missing or has changed since it was indexed; run "
                f"`{command('ingest')}` with the zip's current location",
            )
        )


def _rescan(
    zips: ZipCache,
    media_id: int,
    sha256: str,
    size: int,
    analyzer: FaceAnalyzer | None,
    clip: bool,
) -> tuple[FaceScan | None, ClipPixels | None, Quality | None, bool]:
    """Read one photo back from its zip, then scan it for faces (with ``analyzer``), prepare
    its CLIP input (with ``clip``), and score it. The last value says whether its bytes
    could be read at all."""
    if size > MAX_BUFFERED_IMAGE_BYTES:
        detail = "too large to decode"
        return (
            FaceScan("not_decoded", detail=detail) if analyzer is not None else None,
            ClipPixels("not_decoded", detail=detail) if clip else None,
            None,
            True,
        )
    data = zips.read(media_id, sha256)
    if data is None:
        return None, None, None, False
    try:
        analysis = analyze_image(data, faces=analyzer, clip=clip)
    except UnreadableImageError as exc:
        return (
            FaceScan("failed", detail=str(exc)) if analyzer is not None else None,
            ClipPixels("failed", detail=str(exc)) if clip else None,
            None,
            True,
        )
    scan = analysis.faces
    if analyzer is not None and scan is None:
        scan = FaceScan("failed")
    pixels = analysis.clip
    if clip and pixels is None:
        pixels = ClipPixels("failed")
    return scan, pixels, analysis.quality, True


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
                run.commit()
                conn.execute("BEGIN")
        conn.execute(
            "UPDATE sources SET status = 'complete', completed_at = ? WHERE id = ?",
            (_now(), source_id),
        )
        run.commit()
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
    scan: FaceScan | None = None
    pixels: ClipPixels | None = None
    quality: Quality | None = None
    if member.kind == "video":
        suffix = PurePosixPath(member.path).suffix.lower()
        note = None if video_sample.available() else VIDEO_NOTE
        values.update(kind="video", format=_VIDEO_FORMATS.get(suffix), note=note)
    elif hashed.data is None:
        values.update(kind="image", note="too large to decode; checked for exact duplicates only")
        if run.faces is not None:
            scan = FaceScan("not_decoded", detail="too large to decode")
        pixels = ClipPixels("not_decoded", detail="too large to decode")
    else:
        values["kind"] = "image"
        try:
            analysis = analyze_image(hashed.data, faces=run.faces, clip=run.clip is not None)
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
                exif_modified=exif.modified,
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
            scan = analysis.faces
            pixels = analysis.clip
            quality = analysis.quality
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cursor = run.conn.execute(
        f"INSERT INTO media ({columns}) VALUES ({placeholders})", tuple(values.values())
    )
    assert cursor.lastrowid is not None
    if scan is not None and run.faces is not None:
        store_scan(run.conn, cursor.lastrowid, run.faces.model_id, scan)
        run.faces_scanned += 1
        run.faces_found += len(scan.faces)
    if pixels is not None and run.clip is not None:
        run.clip.add(cursor.lastrowid, pixels)
    if quality is not None:
        store_quality(run.conn, cursor.lastrowid, quality)
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
