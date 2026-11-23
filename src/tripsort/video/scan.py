"""Ingest's video pass, with PyAV (the ``[video]`` extra).

Each video in some zip is copied out to a temporary file (and checked against its SHA-256),
probed for its length, displayed size, and creation time, and, unless it turns out to be a
Live Photo's motion clip, sampled: the faces and the CLIP embedding of each sampled frame are
stored like a photo's, at the frame's time in seconds. One clip at a time, the copy deleted
right after. A video is probed once; it is sampled again when the face model, the CLIP model,
or the sampling changes.
"""

import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from tripsort.clip.preprocess import clip_input
from tripsort.clip.types import ImageEncoder
from tripsort.config import VideoSettings
from tripsort.faces.types import DetectedFace, FaceAnalyzer
from tripsort.index.db import transaction
from tripsort.ingest.sources import ZipCache
from tripsort.ingest.zip_reader import Member, ZipSource
from tripsort.progress import NullProgress, Progress
from tripsort.video.live_photos import live_photo_pairs
from tripsort.video.sample import frames, probe, sample_times

# Changes to how frames are taken (FRAME_MAX_SIDE, turning) must change this.
SAMPLING_VERSION = "upright1280-v1"


def video_scan_id(face_model: str, clip_model: str, settings: VideoSettings) -> str:
    every, most = settings.frame_every_seconds, settings.max_frames
    return f"{face_model}|{clip_model}|every{every:g}s-max{most}-{SAMPLING_VERSION}"


@dataclass
class VideoScanStats:
    probed: int = 0  # videos whose length, size, and date were read
    sampled: int = 0  # videos sampled for faces and CLIP
    frames: int = 0
    live_photos: int = 0  # motion clips found, never sampled
    failed: int = 0  # videos PyAV could not read


def pending_videos(
    conn: sqlite3.Connection, scan_id: str | None
) -> list[tuple[int, str, bool, bool]]:
    """(media id, SHA-256, needs a probe, needs sampling) for every video that needs either."""
    rows = conn.execute(
        "SELECT m.id, m.sha256, m.duration_s IS NULL AND NOT EXISTS ("
        "         SELECT 1 FROM video_scans s WHERE s.media_id = m.id AND s.status = 'failed'), "
        "       ? IS NOT NULL AND NOT EXISTS ("
        "         SELECT 1 FROM video_scans s WHERE s.media_id = m.id AND s.scan_id = ?) "
        "FROM media m WHERE m.kind = 'video' AND m.status = 'ok' "
        "AND m.id IN (SELECT media_id FROM occurrences) ORDER BY m.id",
        (scan_id, scan_id),
    ).fetchall()
    return [(int(r[0]), str(r[1]), bool(r[2]), bool(r[3])) for r in rows if r[2] or r[3]]


def scan_videos(
    conn: sqlite3.Connection,
    settings: VideoSettings,
    analyzer: FaceAnalyzer | None,
    encoder: ImageEncoder | None,
    progress: Progress | None = None,
) -> VideoScanStats:
    """Probe every video not probed yet and, with both models, sample every video not sampled
    with them yet. Commits once per video."""
    progress = progress or NullProgress()
    scan_id = None
    if analyzer is not None and encoder is not None:
        scan_id = video_scan_id(analyzer.model_id, encoder.model_id, settings)
    rows = pending_videos(conn, scan_id)
    stats = VideoScanStats()
    if not rows:
        return stats
    named = live_photo_pairs(conn, float("inf"))  # by name; the length decides below
    with ZipCache(conn) as zips:
        progress.start("videos", len(rows), "files")
        try:
            for media_id, sha256, _, need_sample in rows:
                _scan_one(conn, zips, media_id, sha256, need_sample, named, settings,
                          analyzer, encoder, scan_id, stats)  # fmt: skip
                progress.advance(1)
        finally:
            progress.finish()
    return stats


def _scan_one(
    conn: sqlite3.Connection,
    zips: ZipCache,
    media_id: int,
    sha256: str,
    need_sample: bool,
    named: dict[int, int],
    settings: VideoSettings,
    analyzer: FaceAnalyzer | None,
    encoder: ImageEncoder | None,
    scan_id: str | None,
    stats: VideoScanStats,
) -> None:
    found = zips.find(media_id)
    if found is None:
        return  # the zip is missing or changed; ingest reports it for the photos already
    source, member = found
    try:
        path = _copy_out(source, member, sha256)
    except Exception as exc:  # a zip that cannot be read back never stops the ingest
        _record(conn, media_id, scan_id, "failed", 0, f"could not read it back: {exc}")
        stats.failed += 1
        return
    if path is None:
        _record(conn, media_id, scan_id, "failed", 0, "changed since it was indexed")
        stats.failed += 1
        return
    try:
        info = probe(path)
        with transaction(conn):
            conn.execute(
                "UPDATE media SET duration_s = ?, width = ?, height = ?, video_created = ?, "
                "video_offset = ? WHERE id = ?",
                (info.duration, info.width, info.height, info.created, info.offset, media_id),
            )
            stats.probed += 1
            short = info.duration is None or info.duration <= settings.live_photo_max_seconds
            if media_id in named and short:
                stats.live_photos += 1
                if scan_id is not None:
                    _store_scan(conn, media_id, scan_id, "live_photo", 0, None)
                return
            if not need_sample or scan_id is None or analyzer is None or encoder is None:
                return
            conn.execute(
                "DELETE FROM faces WHERE media_id = ? AND model_id = ? AND frame_ts >= 0",
                (media_id, analyzer.model_id),
            )
            conn.execute(
                "DELETE FROM clip_embeddings WHERE media_id = ? AND model_id = ? AND frame_ts >= 0",
                (media_id, encoder.model_id),
            )
            times = sample_times(info.duration, settings.frame_every_seconds, settings.max_frames)
            taken = 0
            for at, rgb in frames(path, times):
                store_frame_faces(conn, media_id, analyzer.model_id, at, analyzer.analyze(rgb))
                (vector,) = encoder.embed(clip_input(rgb)[np.newaxis])
                conn.execute(
                    "INSERT OR REPLACE INTO clip_embeddings (media_id, frame_ts, model_id, "
                    "embedding) VALUES (?, ?, ?, ?)",
                    (media_id, at, encoder.model_id, np.asarray(vector, np.float32).tobytes()),
                )
                taken += 1
            _store_scan(conn, media_id, scan_id, "ok" if taken else "not_decoded", taken, None)
            stats.sampled += 1
            stats.frames += taken
    except Exception as exc:  # PyAV raises many kinds of errors on broken files
        _record(conn, media_id, scan_id, "failed", 0, f"{type(exc).__name__}: {exc}".strip())
        stats.failed += 1
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def store_frame_faces(
    conn: sqlite3.Connection, media_id: int, model_id: str, at: float, faces: list[DetectedFace]
) -> None:
    conn.executemany(
        "INSERT INTO faces (media_id, model_id, frame_ts, face_index, x, y, w, h, score, "
        "landmarks, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                media_id,
                model_id,
                at,
                index,
                face.x,
                face.y,
                face.w,
                face.h,
                face.score,
                json.dumps([[round(x, 5), round(y, 5)] for x, y in face.landmarks]),
                np.asarray(face.embedding, dtype=np.float32).tobytes(),
            )
            for index, face in enumerate(faces)
        ],
    )


def _store_scan(
    conn: sqlite3.Connection,
    media_id: int,
    scan_id: str,
    status: str,
    taken: int,
    detail: str | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO video_scans (media_id, scan_id, status, frames, detail) "
        "VALUES (?, ?, ?, ?, ?)",
        (media_id, scan_id, status, taken, detail),
    )


def _record(
    conn: sqlite3.Connection,
    media_id: int,
    scan_id: str | None,
    status: str,
    taken: int,
    detail: str,
) -> None:
    """A failure, kept so the video is not read again on every ingest."""
    with transaction(conn):
        _store_scan(conn, media_id, scan_id or "probe", status, taken, detail[:500])


def _copy_out(source: ZipSource, member: Member, sha256: str) -> Path | None:
    """The zip entry copied to a temporary file, or None when its SHA-256 differs."""
    suffix = PurePosixPath(member.path).suffix[:8]
    handle, name = tempfile.mkstemp(prefix="tripsort-video-", suffix=suffix)
    digest = hashlib.sha256()
    try:
        with os.fdopen(handle, "wb") as out, source.open(member) as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except BaseException:
        Path(name).unlink()
        raise
    if digest.hexdigest() != sha256:
        Path(name).unlink()
        return None
    return Path(name)
