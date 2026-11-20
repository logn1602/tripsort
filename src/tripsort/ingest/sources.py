"""Read files back from the zips the index knows, for export, face scans, and review pages."""

import sqlite3
import zipfile
import zlib
from pathlib import Path
from types import TracebackType
from typing import Self

from tripsort.errors import ZipError
from tripsort.ingest.hashing import hash_chunks
from tripsort.ingest.zip_reader import Member, ZipSource


class ZipCache:
    """Opens each source zip at most once and checks it is still the zip that was indexed."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._open: dict[int, tuple[ZipSource, dict[int, Member]] | None] = {}
        self.missing: dict[int, str] = {}  # source id -> zip name, for messages

    def member(self, source_id: int, member_index: int) -> tuple[ZipSource, Member] | None:
        if source_id not in self._open:
            self._open[source_id] = self._load(source_id)
        entry = self._open[source_id]
        if entry is None:
            return None
        source, members = entry
        member = members.get(member_index)
        return (source, member) if member is not None else None

    def find(self, media_id: int) -> tuple[ZipSource, Member] | None:
        """The first readable occurrence of ``media_id``, or None when its zips are gone."""
        for source_id, member_index in self.conn.execute(
            "SELECT source_id, member_index FROM occurrences WHERE media_id = ? "
            "ORDER BY source_id, member_index",
            (media_id,),
        ).fetchall():
            found = self.member(int(source_id), int(member_index))
            if found is not None:
                return found
        return None

    def read(self, media_id: int, sha256: str) -> bytes | None:
        """The bytes of ``media_id``, checked against its SHA-256, or None if unavailable."""
        found = self.find(media_id)
        if found is None:
            return None
        source, member = found
        try:
            hashed = hash_chunks(source.iter_chunks(member), keep=True)
        except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError):
            return None
        return hashed.data if hashed.sha256 == sha256 else None

    def missing_names(self, media_id: int) -> list[str]:
        """The zip names holding ``media_id``, for a message when none of them can be read."""
        rows = self.conn.execute(
            "SELECT DISTINCT o.source_id, s.zip_name FROM occurrences o "
            "JOIN sources s ON s.id = o.source_id WHERE o.media_id = ?",
            (media_id,),
        ).fetchall()
        return sorted({self.missing.get(int(sid), str(name)) for sid, name in rows})

    def _load(self, source_id: int) -> tuple[ZipSource, dict[int, Member]] | None:
        zip_path, zip_name, fingerprint = self.conn.execute(
            "SELECT zip_path, zip_name, fingerprint FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        try:
            source = ZipSource(Path(zip_path))
        except ZipError:
            self.missing[source_id] = str(zip_name)
            return None
        if source.fingerprint != fingerprint:
            source.close()
            self.missing[source_id] = str(zip_name)
            return None
        return source, {member.index: member for member in source.members}

    def close(self) -> None:
        for entry in self._open.values():
            if entry is not None:
                entry[0].close()
        self._open.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
