"""Capture dates: EXIF first, then file-name patterns, then the zip entry's timestamp.

Ingest stores the raw facts (the EXIF time per file; the name and zip time per occurrence)
and ``resolve_dates`` derives ``taken_at`` from them, so the result does not depend on the
order the zips were added, and new patterns apply without reading any zip again.
"""

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

from tripsort.index.db import transaction

_YEARS = range(1990, 2100)


@dataclass(frozen=True)
class NameDate:
    iso: str  # 'YYYY-MM-DDTHH:MM:SS[.fff]', or 'YYYY-MM-DD' when the name has no time
    tz_offset: str | None
    has_time: bool


@dataclass(frozen=True)
class _Pattern:
    regex: re.Pattern[str]
    tz_offset: str | None = None


_PATTERNS = (
    # WhatsApp on phones: IMG-20260714-WA0001.jpg (date only)
    _Pattern(re.compile(r"^(?:IMG|VID)-(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})-WA\d+")),
    # WhatsApp Desktop and Web: "WhatsApp Image 2026-07-14 at 10.23.45.jpeg", maybe "PM"
    _Pattern(
        re.compile(
            r"^WhatsApp (?:Image|Video) (?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2}) at "
            r"(?P<H>\d{1,2})\.(?P<M>\d{2})\.(?P<S>\d{2})(?:\s?(?P<ampm>[AaPp][Mm]))?"
        )
    ),
    # Google Pixel: PXL_20260714_102345123.jpg, in UTC rather than local time
    _Pattern(
        re.compile(
            r"^PXL_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_"
            r"(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})(?P<ms>\d{3})"
        ),
        tz_offset="+00:00",
    ),
    # Generic Android: IMG_20260714_102345.jpg, VID_20260714_102345.mp4
    _Pattern(
        re.compile(
            r"^(?:IMG|VID)_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})"
        )
    ),
    # Samsung: 20260714_102345.jpg
    _Pattern(
        re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})")
    ),
)


def parse_filename_date(name: str) -> NameDate | None:
    """The capture date encoded in a file name, or None if no known pattern matches."""
    for pattern in _PATTERNS:
        match = pattern.regex.match(name)
        if match is None:
            continue
        parts = match.groupdict()
        year, month, day = int(parts["y"]), int(parts["m"]), int(parts["d"])
        if year not in _YEARS:
            continue
        if parts.get("H") is None:
            try:
                return NameDate(datetime(year, month, day).date().isoformat(), None, False)
            except ValueError:
                continue
        hour = int(parts["H"])
        ampm = (parts.get("ampm") or "").lower()
        if ampm:
            if not 1 <= hour <= 12:
                continue
            hour = hour % 12 + (12 if ampm == "pm" else 0)
        try:
            stamp = datetime(year, month, day, hour, int(parts["M"]), int(parts["S"]))
        except ValueError:
            continue
        iso = stamp.isoformat()
        if parts.get("ms"):
            iso += "." + parts["ms"]
        return NameDate(iso, pattern.tz_offset, True)
    return None


@dataclass(frozen=True)
class OccurrenceFacts:
    label: str
    member_path: str
    zip_mtime: str | None


@dataclass(frozen=True)
class ResolvedDate:
    taken_at: str | None
    source: str | None  # 'exif', 'filename', 'zip_mtime'
    tz_offset: str | None


def resolve_date(
    exif_datetime: str | None, exif_offset: str | None, occurrences: list[OccurrenceFacts]
) -> ResolvedDate:
    if exif_datetime is not None:
        return ResolvedDate(exif_datetime, "exif", exif_offset)
    ordered = sorted(occurrences, key=lambda occ: (occ.label, occ.member_path))
    named = [
        found
        for occ in ordered
        if (found := parse_filename_date(PurePosixPath(occ.member_path).name)) is not None
    ]
    if named:
        best = next((found for found in named if found.has_time), named[0])
        return ResolvedDate(best.iso, "filename", best.tz_offset)
    mtimes = sorted(occ.zip_mtime for occ in occurrences if occ.zip_mtime is not None)
    if mtimes:
        return ResolvedDate(mtimes[0], "zip_mtime", None)
    return ResolvedDate(None, None, None)


def resolve_dates(conn: sqlite3.Connection) -> dict[str, int]:
    """Recompute ``taken_at`` for every file. Returns how many dates came from each source."""
    facts: defaultdict[int, list[OccurrenceFacts]] = defaultdict(list)
    for media_id, label, member_path, zip_mtime in conn.execute(
        "SELECT o.media_id, s.label, o.member_path, o.zip_mtime "
        "FROM occurrences o JOIN sources s ON s.id = o.source_id"
    ).fetchall():
        facts[media_id].append(OccurrenceFacts(label, member_path, zip_mtime))
    counts: Counter[str] = Counter()
    updates: list[tuple[str | None, str | None, str | None, int]] = []
    for media_id, exif_datetime, exif_offset, taken_at, source, tz_offset in conn.execute(
        "SELECT id, exif_datetime, exif_offset, taken_at, taken_at_source, tz_offset FROM media"
    ).fetchall():
        resolved = resolve_date(exif_datetime, exif_offset, facts.get(media_id, []))
        counts[resolved.source or "none"] += 1
        if (resolved.taken_at, resolved.source, resolved.tz_offset) != (
            taken_at,
            source,
            tz_offset,
        ):
            updates.append((resolved.taken_at, resolved.source, resolved.tz_offset, media_id))
    if updates:
        with transaction(conn):
            conn.executemany(
                "UPDATE media SET taken_at = ?, taken_at_source = ?, tz_offset = ? WHERE id = ?",
                updates,
            )
    return dict(counts)
