"""Recompressed copies, such as WhatsApp or Telegram copies of a photo.

Two images are copies when their 64-bit pHashes are close and their aspect ratios match.
Burst shots of one moment are often just as close, so EXIF decides:

* Both files have EXIF capture times: the times must match (to the sub-second when both
  record it) and the distance must be at most ``phash_max_distance_both_exif``.
* Otherwise the distance must be at most ``phash_max_distance``.
* Two files that name different camera models, or record different exposure times or ISO
  values, are never copies. Each field counts only when both files record it: a true copy
  keeps the EXIF it has, and a messenger copy has none, so neither is ever split off.
* Two lossless images (PNG) of the same size are never copies: a recompressed copy is lossy,
  and screenshots of one app's layout (tickets, boarding passes) can be as close as copies.

Groups are stars, not connected components. Files are taken in keeper order (most pixels,
then has EXIF, then largest file, then SHA-256), and each one joins the closest compatible
keeper or becomes a keeper itself. A copy that resembles two burst shots therefore cannot
chain them into one group.
"""

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from tripsort.config import Thresholds
from tripsort.index.db import transaction
from tripsort.ingest.hashing import phash_to_int

_BLOCK_ELEMENTS = 4_000_000  # pairwise distances computed per numpy block


@dataclass(frozen=True)
class CopyCandidate:
    media_id: int
    sha256: str
    phash: int
    width: int
    height: int
    has_exif: bool
    bytes: int
    exif_datetime: str | None = None
    camera_model: str | None = None
    format: str | None = None
    exposure_time: float | None = None
    iso: int | None = None

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class CopyGroup:
    keeper: int  # media id
    copies: tuple[tuple[int, int], ...]  # (media id, pHash distance to the keeper)


@dataclass(frozen=True)
class CopyStats:
    groups: int
    copies: int
    bytes_saved: int  # size of the copies left out of all/ (one occurrence each)


def keeper_order(candidate: CopyCandidate) -> tuple[int, int, int, str]:
    """Most pixels, then has EXIF, then largest file; SHA-256 breaks ties."""
    return (-candidate.pixels, -int(candidate.has_exif), -candidate.bytes, candidate.sha256)


def same_capture_time(a: str, b: str) -> bool:
    """Equal to the second, and equal sub-seconds when both record them."""
    if a[:19] != b[:19]:
        return False
    sub_a, sub_b = a[20:], b[20:]
    return not (sub_a and sub_b) or sub_a.ljust(6, "0") == sub_b.ljust(6, "0")


LOSSLESS_FORMATS = frozenset({"PNG"})
EXPOSURE_TOLERANCE = 0.01  # relative: 1/120 s and 1/125 s differ, rounding of one value does not


def cannot_link(a: CopyCandidate, b: CopyCandidate) -> bool:
    """Evidence that two files are different pictures, whatever their pHashes say."""
    if a.camera_model and b.camera_model and a.camera_model != b.camera_model:
        return True
    if a.iso is not None and b.iso is not None and a.iso != b.iso:
        return True
    if (
        a.exposure_time is not None
        and b.exposure_time is not None
        and not math.isclose(a.exposure_time, b.exposure_time, rel_tol=EXPOSURE_TOLERANCE)
    ):
        return True
    if (
        a.format in LOSSLESS_FORMATS
        and b.format in LOSSLESS_FORMATS
        and (a.width, a.height) == (b.width, b.height)
    ):
        return True
    return bool(
        a.exif_datetime
        and b.exif_datetime
        and not same_capture_time(a.exif_datetime, b.exif_datetime)
    )


def pair_limit(a: CopyCandidate, b: CopyCandidate, thresholds: Thresholds) -> int:
    if a.exif_datetime and b.exif_datetime:
        return thresholds.phash_max_distance_both_exif
    return thresholds.phash_max_distance


def find_copy_groups(candidates: list[CopyCandidate], thresholds: Thresholds) -> list[CopyGroup]:
    """Group near-copies; the first member of each group is its keeper."""
    ordered = sorted(candidates, key=keeper_order)
    n = len(ordered)
    if n < 2:
        return []
    hashes = np.array([c.phash for c in ordered], dtype=np.uint64)
    aspect = np.array([c.width / c.height for c in ordered], dtype=np.float64)
    max_any = max(thresholds.phash_max_distance, thresholds.phash_max_distance_both_exif)

    # For each file j, the higher-priority files i < j within the widest threshold.
    neighbors: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    rows_per_block = max(1, _BLOCK_ELEMENTS // n)
    for start in range(1, n, rows_per_block):
        stop = min(n, start + rows_per_block)
        rows = np.arange(start, stop)
        distances = np.bitwise_count(hashes[start:stop, None] ^ hashes[None, :stop])
        mask = (distances <= max_any) & (np.arange(stop)[None, :] < rows[:, None])
        j_idx, i_idx = np.nonzero(mask)
        if j_idx.size == 0:
            continue
        j_abs = j_idx + start
        a_i, a_j = aspect[i_idx], aspect[j_abs]
        aspect_ok = np.abs(a_i - a_j) / np.maximum(a_i, a_j) <= thresholds.copy_max_aspect_diff
        for j, i, d in zip(
            j_abs[aspect_ok].tolist(),
            i_idx[aspect_ok].tolist(),
            distances[j_idx, i_idx][aspect_ok].tolist(),
            strict=True,
        ):
            neighbors[j].append((d, i))

    keeper_of: dict[int, int] = {}  # copy index -> keeper index
    members: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)  # keeper -> copies
    for j in range(1, n):
        best: tuple[int, int] | None = None
        for distance, i in sorted(neighbors.get(j, ())):
            if i in keeper_of:  # only keepers take copies; copies never chain
                continue
            a, b = ordered[i], ordered[j]
            if distance > pair_limit(a, b, thresholds) or cannot_link(a, b):
                continue
            if any(cannot_link(ordered[k], b) for k, _ in members[i]):
                continue
            best = (distance, i)
            break
        if best is not None:
            keeper_of[j] = best[1]
            members[best[1]].append((j, best[0]))

    return [
        CopyGroup(
            keeper=ordered[i].media_id,
            copies=tuple((ordered[k].media_id, d) for k, d in copies),
        )
        for i, copies in sorted(members.items())
    ]


def load_candidates(conn: sqlite3.Connection) -> list[CopyCandidate]:
    rows = conn.execute(
        "SELECT id, sha256, phash, width, height, has_exif, bytes, exif_datetime, camera_model, "
        "format, exposure_time, iso FROM media "
        "WHERE status = 'ok' AND kind = 'image' AND phash IS NOT NULL "
        "AND width > 0 AND height > 0"
    ).fetchall()
    return [
        CopyCandidate(
            media_id=int(row[0]),
            sha256=str(row[1]),
            phash=phash_to_int(row[2]),
            width=int(row[3]),
            height=int(row[4]),
            has_exif=bool(row[5]),
            bytes=int(row[6]),
            exif_datetime=row[7],
            camera_model=row[8],
            format=row[9],
            exposure_time=row[10],
            iso=row[11],
        )
        for row in rows
    ]


def rebuild_copy_groups(conn: sqlite3.Connection, thresholds: Thresholds) -> CopyStats:
    """Recompute every copy group from the index (deterministic, pixels never touched)."""
    candidates = load_candidates(conn)
    groups = find_copy_groups(candidates, thresholds)
    sizes = {c.media_id: c.bytes for c in candidates}
    with transaction(conn):
        conn.execute("DELETE FROM media_groups WHERE kind = 'copy'")
        for group in groups:
            cursor = conn.execute("INSERT INTO media_groups (kind) VALUES ('copy')")
            group_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO group_members (group_id, media_id, is_keeper) VALUES (?, ?, 1)",
                (group_id, group.keeper),
            )
            conn.executemany(
                "INSERT INTO group_members (group_id, media_id, is_keeper, distance) "
                "VALUES (?, ?, 0, ?)",
                [(group_id, media_id, distance) for media_id, distance in group.copies],
            )
    copies = [media_id for group in groups for media_id, _ in group.copies]
    return CopyStats(
        groups=len(groups),
        copies=len(copies),
        bytes_saved=sum(sizes[media_id] for media_id in copies),
    )
