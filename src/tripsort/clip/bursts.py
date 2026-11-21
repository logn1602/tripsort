"""Bursts: several shots of one moment, from one camera, seconds apart and nearly alike.

Kept photos are put in time order per camera. A burst starts at a photo and takes each
later shot that is at most ``burst_window_seconds`` after the burst's first shot and at
least ``burst_similarity`` similar to it (CLIP). Like copy groups, a burst is a star around
its first shot, so a slow pan cannot chain a whole afternoon together. A camera is the EXIF
make and model, or, for photos without a camera model, the zip they came from. Only times
from EXIF or the file name count; a zip entry's time is when the file was zipped.

Every member of a burst stays in all/; a later release picks the best shot of each.
"""

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from tripsort.clip.types import CLIP_MODEL_ID, EMBEDDING_SIZE
from tripsort.config import Thresholds
from tripsort.dedupe.copies import kept_images

TRUSTED_TIMES = ("exif", "filename")


@dataclass(frozen=True)
class Shot:
    media_id: int
    camera: str
    seconds: float  # capture time, as seconds on that camera's clock
    embedding: npt.NDArray[np.float32]


@dataclass(frozen=True)
class BurstStats:
    bursts: int = 0
    photos: int = 0  # photos in a burst


def group_bursts(shots: list[Shot], min_similarity: float, window: float) -> list[list[int]]:
    """The bursts among ``shots``, as lists of media ids in time order (two shots or more)."""
    by_camera: defaultdict[str, list[Shot]] = defaultdict(list)
    for shot in shots:
        by_camera[shot.camera].append(shot)
    bursts: list[list[int]] = []
    for camera in sorted(by_camera):
        ordered = sorted(by_camera[camera], key=lambda s: (s.seconds, s.media_id))
        taken: set[int] = set()
        for i, first in enumerate(ordered):
            if first.media_id in taken:
                continue
            members = [first.media_id]
            for later in ordered[i + 1 :]:
                if later.seconds - first.seconds > window:
                    break
                if later.media_id in taken:
                    continue
                if float(first.embedding @ later.embedding) >= min_similarity:
                    members.append(later.media_id)
            if len(members) > 1:
                bursts.append(members)
                taken.update(members)
    return bursts


def analyze_bursts(
    conn: sqlite3.Connection, thresholds: Thresholds, model_id: str = CLIP_MODEL_ID
) -> BurstStats:
    """Rebuild the burst groups from the kept photos. Call inside a transaction."""
    conn.execute("DELETE FROM media_groups WHERE kind = 'burst'")
    kept = kept_images(conn)
    placeholders = ", ".join("?" for _ in TRUSTED_TIMES)
    shots = []
    for media_id, taken_at, make, model, source, blob in conn.execute(
        "SELECT m.id, m.taken_at, m.camera_make, m.camera_model, "
        "       (SELECT MIN(o.source_id) FROM occurrences o WHERE o.media_id = m.id), "
        "       e.embedding "
        "FROM media m JOIN clip_embeddings e ON e.media_id = m.id AND e.model_id = ? "
        "AND e.frame_ts = -1 WHERE m.taken_at IS NOT NULL "
        f"AND m.taken_at_source IN ({placeholders})",
        (model_id, *TRUSTED_TIMES),
    ):
        if int(media_id) not in kept or (seconds := _seconds(str(taken_at))) is None:
            continue
        camera = f"camera:{make or ''}|{model}" if model else f"zip:{source}"
        vector = np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE)
        shots.append(Shot(int(media_id), camera, seconds, vector))
    bursts = group_bursts(shots, thresholds.burst_similarity, thresholds.burst_window_seconds)
    for members in bursts:  # in camera, then time order: stable group ids for the same input
        cursor = conn.execute("INSERT INTO media_groups (kind) VALUES ('burst')")
        conn.executemany(
            "INSERT INTO group_members (group_id, media_id) VALUES (?, ?)",
            [(cursor.lastrowid, media_id) for media_id in members],
        )
    return BurstStats(bursts=len(bursts), photos=sum(len(b) for b in bursts))


_EPOCH = datetime(1970, 1, 1)


def _seconds(taken_at: str) -> float | None:
    """The camera clock's time as seconds, ignoring any zone: shots are only ever compared
    with shots from the same camera, and a local time has no daylight-saving jumps here."""
    try:
        stamp = datetime.fromisoformat(taken_at)
    except ValueError:
        return None
    return (stamp.replace(tzinfo=None) - _EPOCH).total_seconds()
