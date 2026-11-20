"""Write face scans into the index."""

import json
import sqlite3

import numpy as np

from tripsort.faces.types import FaceScan


def store_scan(conn: sqlite3.Connection, media_id: int, model_id: str, scan: FaceScan) -> None:
    """Record that ``media_id`` was scanned with ``model_id``, and every face found."""
    conn.execute(
        "INSERT INTO face_scans (media_id, model_id, status, faces, detail) VALUES (?, ?, ?, ?, ?)",
        (media_id, model_id, scan.status, len(scan.faces), scan.detail),
    )
    conn.executemany(
        "INSERT INTO faces (media_id, model_id, face_index, x, y, w, h, score, landmarks, "
        "embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                media_id,
                model_id,
                index,
                face.x,
                face.y,
                face.w,
                face.h,
                face.score,
                json.dumps([[round(x, 5), round(y, 5)] for x, y in face.landmarks]),
                np.asarray(face.embedding, dtype=np.float32).tobytes(),
            )
            for index, face in enumerate(scan.faces)
        ],
    )


def pending_scans(conn: sqlite3.Connection, model_id: str) -> int:
    """Photos (decodable kinds, in some zip) not scanned for faces with ``model_id`` yet."""
    (count,) = conn.execute(
        "SELECT COUNT(*) FROM media m WHERE m.kind = 'image' AND m.status = 'ok' "
        "AND m.id IN (SELECT media_id FROM occurrences) "
        "AND NOT EXISTS (SELECT 1 FROM face_scans s WHERE s.media_id = m.id AND s.model_id = ?)",
        (model_id,),
    ).fetchone()
    return int(count)
