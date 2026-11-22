"""``manifest.json``: every file, where it came from, and what tripsort decided about it.

It holds no absolute paths (zips appear by file name only), no GPS, no timestamps, and no
face boxes or embeddings, so an export with the same inputs and config writes the same bytes
every time. It does name the people in each photo, as the people/ folders do. Each photo's
category shares are rounded to four decimals, and bursts are numbered from 1 in camera and
time order.
"""

import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tripsort import __version__
from tripsort.clip.store import pending_clip_scans
from tripsort.clip.types import CLIP_MODEL_ID
from tripsort.config import Config
from tripsort.cull.quality import QUALITY_VERSION
from tripsort.export.writer import TMP_SUFFIX
from tripsort.faces.names import display_name
from tripsort.faces.people import person_infos
from tripsort.faces.store import pending_scans
from tripsort.faces.types import FACE_MODEL_ID
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

    people_in: defaultdict[int, list[str]] = defaultdict(list)
    for media_id, number, name in conn.execute(
        "SELECT DISTINCT f.media_id, p.number, p.name FROM faces f "
        "JOIN people p ON p.id = f.person_id WHERE f.match IS NOT NULL"
    ):
        people_in[int(media_id)].append(display_name(int(number), name))
    face_counts = {
        int(media_id): {"foreground": int(fg), "background": int(bg)}
        for media_id, fg, bg in conn.execute(
            "SELECT s.media_id, "
            "  (SELECT COUNT(*) FROM faces f WHERE f.media_id = s.media_id "
            "   AND f.model_id = s.model_id AND f.is_foreground = 1), "
            "  (SELECT COUNT(*) FROM faces f WHERE f.media_id = s.media_id "
            "   AND f.model_id = s.model_id AND f.is_foreground = 0 AND f.score >= ?) "
            "FROM face_scans s WHERE s.model_id = ? AND s.status = 'ok'",
            (config.people.min_face_score, FACE_MODEL_ID),
        )
    }

    scores: defaultdict[int, dict[str, float]] = defaultdict(dict)
    primary: dict[int, str] = {}
    for media_id, label, score, is_primary in conn.execute(
        "SELECT media_id, label, score, is_primary FROM categories ORDER BY media_id"
    ):
        scores[int(media_id)][str(label)] = round(float(score), 4)
        if is_primary:
            primary[int(media_id)] = str(label)
    burst_of: dict[int, int] = {}
    best_of: dict[int, bool] = {}  # burst shots: whether it is the burst's best
    for number, (group_id,) in enumerate(
        conn.execute("SELECT id FROM media_groups WHERE kind = 'burst' ORDER BY id").fetchall(), 1
    ):
        for media_id, is_keeper in conn.execute(
            "SELECT media_id, is_keeper FROM group_members WHERE group_id = ?", (group_id,)
        ):
            burst_of[int(media_id)] = number
            best_of[int(media_id)] = bool(is_keeper)
    quality: dict[int, dict[str, float]] = {}
    reject: dict[int, str] = {}
    for media_id, sharpness, sharp_tiles, luma, dark, bright, reason in conn.execute(
        "SELECT media_id, sharpness, sharp_tiles, luma_mean, dark_frac, bright_frac, reject "
        "FROM quality"
    ):
        quality[int(media_id)] = {
            "sharpness": round(float(sharpness), 1),
            "sharp_tiles": round(float(sharp_tiles), 1),
            "luma_mean": round(float(luma), 1),
            "dark_frac": round(float(dark), 4),
            "bright_frac": round(float(bright), 4),
        }
        if reason is not None:
            reject[int(media_id)] = str(reason)
    sha_of = {int(i): str(s) for i, s in conn.execute("SELECT id, sha256 FROM media")}
    frames_of = {
        int(media_id): int(frames)
        for media_id, frames in conn.execute(
            "SELECT media_id, MAX(frames) FROM video_scans WHERE status = 'ok' GROUP BY media_id"
        )
    }
    motion_of: dict[int, str] = {}  # motion clip -> its still's SHA-256
    motion_clip: dict[int, str] = {}  # still -> its motion clip's SHA-256
    for motion, still in conn.execute("SELECT motion_id, still_id FROM live_photos"):
        motion_of[int(motion)] = sha_of[int(still)]
        motion_clip[int(still)] = sha_of[int(motion)]

    files = []
    for row in conn.execute(
        "SELECT id, sha256, kind, format, bytes, width, height, taken_at, tz_offset, "
        "taken_at_source, camera_make, camera_model, lens_model, phash, status, note, "
        "shot, is_selfie, duration_s FROM media ORDER BY COALESCE(taken_at, ''), sha256"
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
                "people": sorted(people_in.get(media_id, [])),
                "shot": row["shot"],
                "selfie": None if row["is_selfie"] is None else bool(row["is_selfie"]),
                "faces": face_counts.get(media_id),
                "category": primary.get(media_id),
                "category_scores": scores.get(media_id),
                "burst": burst_of.get(media_id),
                "burst_best": best_of.get(media_id),
                "quality": quality.get(media_id),
                "reject": reject.get(media_id),
                "motion_clip": motion_clip.get(media_id),
                "motion_of": motion_of.get(media_id),
                "duration_s": row["duration_s"],
                "frames": frames_of.get(media_id),
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

    people = sorted(
        (p for p in person_infos(conn) if p.photos or p.named_by),
        key=lambda p: (-p.photos, p.number),
    )
    shots = dict.fromkeys(("solo", "group", "none", "selfie"), 0)
    for shot, n in conn.execute(
        "SELECT shot, COUNT(*) FROM media WHERE shot IS NOT NULL GROUP BY shot"
    ):
        shots[str(shot)] = int(n)
    (shots["selfie"],) = conn.execute("SELECT COUNT(*) FROM media WHERE is_selfie = 1").fetchone()

    return {
        "manifest_version": MANIFEST_VERSION,
        "tripsort_version": __version__,
        "config_hash": config.hash,
        "thresholds": asdict(config.thresholds),
        "people_settings": asdict(config.people),
        "categories": {name: list(prompts) for name, prompts in config.categories.items()},
        "analyzers": {
            "phash": PHASH_ALGORITHM,
            "faces": FACE_MODEL_ID,
            "clip": CLIP_MODEL_ID,
            "quality": QUALITY_VERSION,
        },
        "people": [
            {"name": p.name, "photos": p.photos, "solo": p.solo, "named_by": p.named_by}
            for p in people
        ],
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
            "people": sum(1 for p in people if p.photos),
            "named_people": sum(1 for p in people if p.photos and p.named_by),
            "shots": shots,
            "faces_not_scanned": pending_scans(conn, FACE_MODEL_ID),
            "categories": {
                name: sum(1 for label in primary.values() if label == name)
                for name in config.categories
            },
            "bursts": len(set(burst_of.values())),
            "photos_in_bursts": len(burst_of),
            "best_shots": sum(best_of.values()),
            "rejects": {
                reason: sum(1 for r in reject.values() if r == reason)
                for reason in ("dark", "blown_out", "blurry")
            },
            "live_photos": len(motion_of),
            "clip_not_embedded": pending_clip_scans(conn, CLIP_MODEL_ID),
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
