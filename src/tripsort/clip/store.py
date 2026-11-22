"""Write CLIP embeddings into the index, a batch of photos at a time."""

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from tripsort.clip.types import BATCH_SIZE, ClipInput, ClipPixels, ImageEncoder


@dataclass
class ClipBatch:
    """Photos waiting for the image encoder. ``add`` runs the encoder once ``BATCH_SIZE``
    photos are waiting; call ``flush`` before every commit, so a photo's embedding is
    committed with it."""

    conn: sqlite3.Connection
    encoder: ImageEncoder
    embedded: int = 0  # photos with an embedding, this run
    _waiting: list[tuple[int, ClipInput]] = field(default_factory=list)

    def add(self, media_id: int, pixels: ClipPixels) -> None:
        if pixels.status != "ok" or pixels.input is None:
            self._scan(
                media_id, pixels.status if pixels.status != "ok" else "failed", pixels.detail
            )
            return
        self._waiting.append((media_id, pixels.input))
        if len(self._waiting) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._waiting:
            return
        waiting, self._waiting = self._waiting, []
        try:
            vectors = self.encoder.embed(np.stack([pixels for _, pixels in waiting]))
        except Exception as exc:  # an encoder failure never stops the ingest
            detail = f"{type(exc).__name__}: {exc}".strip()
            for media_id, _ in waiting:
                self._scan(media_id, "failed", detail)
            return
        model_id = self.encoder.model_id
        self.conn.executemany(
            "INSERT OR REPLACE INTO clip_embeddings (media_id, model_id, embedding) "
            "VALUES (?, ?, ?)",
            [
                (media_id, model_id, np.asarray(vector, np.float32).tobytes())
                for (media_id, _), vector in zip(waiting, vectors, strict=True)
            ],
        )
        for media_id, _ in waiting:
            self._scan(media_id, "ok", None)
        self.embedded += len(waiting)

    def _scan(self, media_id: int, status: str, detail: str | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO clip_scans (media_id, model_id, status, detail) "
            "VALUES (?, ?, ?, ?)",
            (media_id, self.encoder.model_id, status, detail),
        )


def pending_clip_scans(conn: sqlite3.Connection, model_id: str) -> int:
    """Photos (decodable kinds, in some zip) not embedded with ``model_id`` yet."""
    (count,) = conn.execute(
        "SELECT COUNT(*) FROM media m WHERE m.kind = 'image' AND m.status = 'ok' "
        "AND m.id IN (SELECT media_id FROM occurrences) "
        "AND NOT EXISTS (SELECT 1 FROM clip_scans s WHERE s.media_id = m.id AND s.model_id = ?)",
        (model_id,),
    ).fetchone()
    return int(count)
