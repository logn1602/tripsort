"""Live Photos: a short video beside a still of the same name belongs to that still.

Phones and photo exports keep a Live Photo as a still and a short clip with the same name:
IMG_1234.HEIC with IMG_1234.MOV from an iPhone, or with IMG_1234.MP4 in a Google Photos export.
A video whose name, without its extension, matches a photo's in the same folder of the same zip
(ignoring case) is that photo's motion clip, unless its length is known (with PyAV) and longer
than ``live_photo_max_seconds``. A motion clip is never sampled as a video, and export puts it
next to its still in all/ only.
"""

import sqlite3
from pathlib import PurePosixPath

from tripsort.config import VideoSettings


def live_photo_pairs(conn: sqlite3.Connection, max_seconds: float) -> dict[int, int]:
    """motion clip media id -> its still's media id."""
    stills: dict[tuple[int, str, str], int] = {}
    videos: list[tuple[int, tuple[int, str, str], float | None]] = []
    for media_id, source_id, member_path, kind, duration in conn.execute(
        "SELECT o.media_id, o.source_id, o.member_path, m.kind, m.duration_s "
        "FROM occurrences o JOIN media m ON m.id = o.media_id WHERE m.status = 'ok' "
        "ORDER BY o.source_id, o.member_index"
    ):
        path = PurePosixPath(str(member_path))
        key = (int(source_id), str(path.parent).casefold(), path.stem.casefold())
        if kind == "image":
            stills.setdefault(key, int(media_id))
        else:
            videos.append((int(media_id), key, None if duration is None else float(duration)))
    pairs: dict[int, int] = {}
    for media_id, key, duration in videos:
        if key in stills and (duration is None or duration <= max_seconds):
            pairs.setdefault(media_id, stills[key])
    return pairs


def analyze_live_photos(conn: sqlite3.Connection, settings: VideoSettings) -> int:
    """Store every Live Photo pair. Call inside a transaction; returns how many pairs."""
    conn.execute("DELETE FROM live_photos")
    pairs = live_photo_pairs(conn, settings.live_photo_max_seconds)
    conn.executemany(
        "INSERT INTO live_photos (motion_id, still_id) VALUES (?, ?)", sorted(pairs.items())
    )
    return len(pairs)
