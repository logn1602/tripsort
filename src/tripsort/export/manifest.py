"""``manifest.json``: every file, where it came from, and what tripsort decided about it.

It holds no absolute paths (zips appear by file name only), no GPS, and no timestamps, so
an export with the same inputs and config writes the same bytes every time.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tripsort import __version__
from tripsort.config import Config
from tripsort.export.writer import TMP_SUFFIX
from tripsort.ingest.hashing import PHASH_ALGORITHM
from tripsort.ingest.zip_reader import SKIP_JUNK, skip_hint
from tripsort.stats import ProjectStats

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


def build_manifest(conn: sqlite3.Connection, config: Config, stats: ProjectStats) -> dict[str, Any]:
    paths: defaultdict[int, list[str]] = defaultdict(list)
    for media_id, relpath in conn.execute(
        "SELECT media_id, relpath FROM exported_files ORDER BY relpath"
    ):
        paths[int(media_id)].append(str(relpath))

    copy_of: dict[int, tuple[str, int | None]] = {}
    for media_id, keeper_sha, distance in conn.execute(
        "SELECT gm.media_id, km.sha256, gm.distance FROM group_members gm "
        "JOIN media_groups g ON g.id = gm.group_id AND g.kind = 'copy' "
        "JOIN group_members k ON k.group_id = gm.group_id AND k.is_keeper = 1 "
        "JOIN media km ON km.id = k.media_id WHERE gm.is_keeper = 0"
    ):
        copy_of[int(media_id)] = (str(keeper_sha), distance)

    occurrences: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for media_id, label, member_path, member_raw in conn.execute(
        "SELECT o.media_id, s.label, o.member_path, o.member_raw "
        "FROM occurrences o JOIN sources s ON s.id = o.source_id "
        "ORDER BY s.label, o.member_path, o.member_index"
    ):
        occurrences[int(media_id)].append(
            {
                "source": label,
                "member": member_path,
                "member_raw": bytes(member_raw).hex() if member_raw is not None else None,
            }
        )

    files = []
    for row in conn.execute(
        "SELECT id, sha256, kind, format, bytes, width, height, taken_at, tz_offset, "
        "taken_at_source, camera_make, camera_model, lens_model, phash, status, note "
        "FROM media ORDER BY COALESCE(taken_at, ''), sha256"
    ):
        media_id = int(row["id"])
        if media_id not in occurrences:
            continue
        keeper_sha, distance = copy_of.get(media_id, (None, None))
        camera = {
            "make": row["camera_make"],
            "model": row["camera_model"],
            "lens": row["lens_model"],
        }
        files.append(
            {
                "sha256": row["sha256"],
                "kind": row["kind"],
                "format": row["format"],
                "bytes": row["bytes"],
                "width": row["width"],
                "height": row["height"],
                "taken_at": row["taken_at"],
                "tz_offset": row["tz_offset"],
                "date_source": row["taken_at_source"],
                "camera": camera if any(camera.values()) else None,
                "phash": row["phash"],
                "status": row["status"],
                "note": row["note"],
                "keeper": row["status"] == "ok" and keeper_sha is None,
                "copy_of": keeper_sha,
                "copy_distance": distance,
                "paths": paths.get(media_id, []),
                "occurrences": occurrences[media_id],
            }
        )

    skipped = [
        {
            "source": label,
            "member": member_path,
            "member_raw": bytes(member_raw).hex() if member_raw is not None else None,
            "bytes": size,
            "reason": reason,
            "detail": detail,
            "hint": skip_hint(reason, detail),
        }
        for label, member_path, member_raw, size, reason, detail in conn.execute(
            "SELECT s.label, k.member_path, k.member_raw, k.bytes, k.reason, k.detail "
            "FROM skipped_members k JOIN sources s ON s.id = k.source_id "
            "WHERE k.reason != ? ORDER BY s.label, k.member_path, k.member_index",
            (SKIP_JUNK,),
        )
    ]

    sources = [
        {
            "label": label,
            "zip": zip_name,
            "fingerprint": fingerprint,
            "members": members,
            "status": status,
        }
        for label, zip_name, fingerprint, members, status in conn.execute(
            "SELECT label, zip_name, fingerprint, members, status FROM sources ORDER BY label"
        )
    ]

    return {
        "manifest_version": MANIFEST_VERSION,
        "tripsort_version": __version__,
        "config_hash": config.hash,
        "thresholds": asdict(config.thresholds),
        "analyzers": {"phash": PHASH_ALGORITHM},
        "summary": {
            "sources": stats.sources,
            "files_found": stats.files_found,
            "unique": stats.unique,
            "exact_duplicates": stats.exact_duplicates,
            "near_copies": stats.near_copies,
            "copy_groups": stats.copy_groups,
            "kept": stats.kept,
            "images": stats.images,
            "videos": stats.videos,
            "not_analyzed": stats.not_analyzed,
            "unreadable": stats.unreadable,
            "skipped": stats.skipped,
            "bytes_in": stats.bytes_in,
            "bytes_kept": stats.bytes_kept,
            "bytes_saved": stats.bytes_saved,
            "exact_bytes_saved": stats.exact_bytes_saved,
            "copy_bytes_saved": stats.copy_bytes_saved,
        },
        "sources": sources,
        "files": files,
        "skipped": skipped,
    }


def write_manifest(
    conn: sqlite3.Connection, project_dir: Path, config: Config, stats: ProjectStats
) -> Path:
    """Write ``manifest.json`` atomically (temporary file, then rename).

    An unchanged manifest is left as it is, so a re-run with nothing new writes nothing
    and sync tools have nothing to upload.
    """
    text = json.dumps(build_manifest(conn, config, stats), indent=2, ensure_ascii=False) + "\n"
    path = project_dir / MANIFEST_NAME
    data = text.encode("utf-8")
    try:
        if path.stat().st_size == len(data) and path.read_bytes() == data:
            return path
    except FileNotFoundError:
        pass
    tmp = path.with_name(f".{MANIFEST_NAME}{TMP_SUFFIX}")
    with tmp.open("w", encoding="utf-8", newline="\n") as fp:
        fp.write(text)
    tmp.replace(path)
    return path
