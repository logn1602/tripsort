"""Turn the stored faces into people and shot types. Reads and writes the index only.

The steps, in order:

1. the foreground filter, per photo;
2. clustering of the foreground faces of kept photos (a copy would count a face twice);
3. faces unlike their cluster's average face taken out, and unclustered foreground faces
   joining the most similar person, if close enough (see ``cluster``);
4. carrying people over from the last analysis (see ``identity``);
5. names from refs/;
6. background faces of named people, under a stricter threshold;
7. shot types and selfies;
8. merge suggestions: unnamed people who may be a named person split in two (see
   ``suggest``). Nothing is merged; the user decides.
"""

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tripsort.config import PeopleSettings
from tripsort.faces.cluster import NOISE, attach_noise, cluster_embeddings, eject_outliers
from tripsort.faces.foreground import foreground
from tripsort.faces.identity import carry_over
from tripsort.faces.names import display_name, name_key
from tripsort.faces.naming import load_refs, match_refs
from tripsort.faces.shots import is_front_camera, is_selfie, shot_type
from tripsort.faces.suggest import Suggestion, suggest_merges
from tripsort.faces.types import EMBEDDING_SIZE, FACE_MODEL_ID, normalized
from tripsort.index.db import get_meta, set_meta

NEXT_NUMBER_KEY = "next_person_number"
SHOT_KINDS = ("solo", "group", "none", "selfie")


@dataclass(frozen=True)
class PeopleStats:
    people: int = 0  # people in at least one photo
    named: int = 0  # of those, people with a name
    photos_with_people: int = 0
    foreground_faces: int = 0  # in kept photos
    clustered: int = 0
    attached: int = 0  # unclustered foreground faces that joined a close person
    background: int = 0  # background faces matched to a named person
    unassigned: int = 0  # foreground faces in no person's folder
    shots: dict[str, int] = field(default_factory=dict)  # solo, group, none, selfie -> photos
    not_scanned: int = 0  # kept photos without a face scan yet
    joined: list[tuple[str, ...]] = field(default_factory=list)  # confirmed people kept apart
    refs_unmatched: list[str] = field(default_factory=list)  # ref names that named nobody
    # unnamed people who may be a named person split in two, most similar first
    suggestions: list[Suggestion] = field(default_factory=list)


def analyze_people(
    conn: sqlite3.Connection, settings: PeopleSettings, model_id: str = FACE_MODEL_ID
) -> PeopleStats:
    """Run every step and store the results. Call inside a transaction."""
    kept = _kept_images(conn)
    scanned = {
        int(media_id)
        for (media_id,) in conn.execute(
            "SELECT media_id FROM face_scans WHERE model_id = ? AND status = 'ok'", (model_id,)
        )
    }
    rows = conn.execute(
        "SELECT f.id, f.media_id, f.h, f.score, f.embedding, f.person_id FROM faces f "
        "JOIN media m ON m.id = f.media_id WHERE f.model_id = ? AND f.frame_ts = -1 "
        "ORDER BY m.sha256, f.face_index",
        (model_id,),
    ).fetchall()
    n = len(rows)
    face_ids = [int(r[0]) for r in rows]
    media = np.array([int(r[1]) for r in rows], dtype=np.int64)
    heights = np.array([float(r[2]) for r in rows], dtype=np.float64)
    scores = np.array([float(r[3]) for r in rows], dtype=np.float64)
    emb = (
        np.frombuffer(b"".join(bytes(r[4]) for r in rows), dtype=np.float32).reshape(
            n, EMBEDDING_SIZE
        )
        if n
        else np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    )
    previous = [int(r[5]) if r[5] is not None else None for r in rows]

    # 1. foreground filter
    counted = np.zeros(n, dtype=np.bool_)
    fg = np.zeros(n, dtype=np.bool_)
    by_photo: defaultdict[int, list[int]] = defaultdict(list)
    for i, media_id in enumerate(media):
        by_photo[int(media_id)].append(i)
    for idx in by_photo.values():
        counted[idx], fg[idx] = foreground(
            heights[idx],
            scores[idx],
            min_score=settings.min_face_score,
            min_ratio=settings.foreground_min_ratio,
            min_height=settings.foreground_min_height,
        )
    in_kept = np.array([int(m) in kept for m in media], dtype=np.bool_)

    # 2-4. cluster, attach, carry over
    candidates = np.flatnonzero(fg & in_kept)
    vectors = emb[candidates]
    labels = cluster_embeddings(vectors, settings)
    labels = eject_outliers(
        vectors, labels, settings.member_min_similarity, settings.min_cluster_size
    )
    labels, attached = attach_noise(
        vectors, labels, media[candidates], settings.attach_min_similarity
    )
    pinned = {int(pid) for (pid,) in conn.execute("SELECT id FROM people WHERE pinned = 1")}
    carry = carry_over(
        labels,
        [previous[i] for i in candidates],
        pinned,
        vectors,
        min_faces=settings.min_cluster_size,
    )
    sizes = Counter(int(label) for label in labels if label != NOISE)
    new_labels = sorted(
        {
            int(labels[k])
            for k in range(len(candidates))
            if labels[k] != NOISE
            and k not in carry.face_person
            and carry.cluster_person.get(int(labels[k])) is None
        },
        key=lambda label: (-sizes[label], label),
    )
    new_person = dict(zip(new_labels, _create_people(conn, len(new_labels)), strict=True))

    person: list[int | None] = [None] * n
    match: list[str | None] = [None] * n
    cluster: list[int | None] = [None] * n
    for k, i in enumerate(candidates.tolist()):
        label = int(labels[k])
        cluster[i] = label
        if label == NOISE:
            continue
        pid = carry.person_of(k, label)
        person[i] = pid if pid is not None else new_person[label]
        match[i] = "attached" if attached[k] else "cluster"
    members: defaultdict[int, list[int]] = defaultdict(list)
    for i, pid in enumerate(person):
        if pid is not None:
            members[pid].append(i)
    centroid = {pid: normalized(emb[idx].mean(axis=0)) for pid, idx in members.items()}

    # 5. names from refs
    people_rows = _people(conn)
    user_named = {
        pid: str(row["name"])
        for pid, row in people_rows.items()
        if row["named_by"] == "user" and row["name"]
    }
    refs = load_refs(conn, model_id)
    matches = match_refs(refs, centroid, user_named, settings.ref_min_similarity)
    _apply_ref_names(conn, people_rows, matches.matched)
    people_rows = _people(conn)

    # 6. background faces of named people
    named_ids = [pid for pid in sorted(centroid) if people_rows[pid]["name"]]
    background = 0
    if named_ids:
        matrix = np.stack([centroid[pid] for pid in named_ids])
        in_photo: defaultdict[int, set[int]] = defaultdict(set)
        for i, pid in enumerate(person):
            if pid is not None:
                in_photo[int(media[i])].add(pid)
        faces = np.flatnonzero(counted & ~fg & in_kept)
        similarities = emb[faces] @ matrix.T
        for row, i in enumerate(faces.tolist()):
            for column in np.argsort(-similarities[row], kind="stable"):
                if similarities[row, column] < settings.background_min_similarity:
                    break
                pid = named_ids[int(column)]
                if pid not in in_photo[int(media[i])]:
                    person[i], match[i] = pid, "background"
                    in_photo[int(media[i])].add(pid)
                    background += 1
                    break

    # store faces and people
    conn.execute("UPDATE faces SET match = NULL, cluster = NULL WHERE model_id != ?", (model_id,))
    conn.executemany(
        "UPDATE faces SET is_foreground = ?, cluster = ?, person_id = COALESCE(?, person_id), "
        "match = ? WHERE id = ?",
        [(int(fg[i]), cluster[i], person[i], match[i], face_ids[i]) for i in range(n)],
    )
    conn.execute("UPDATE people SET centroid = NULL")
    conn.executemany(
        "UPDATE people SET centroid = ? WHERE id = ?",
        [(vector.astype(np.float32).tobytes(), pid) for pid, vector in centroid.items()],
    )
    conn.execute(
        "DELETE FROM people WHERE name IS NULL AND pinned = 0 AND id NOT IN "
        "(SELECT person_id FROM faces WHERE person_id IS NOT NULL)"
    )

    # 7. shot types
    fg_count = Counter(int(media[i]) for i in np.flatnonzero(fg))
    shots: Counter[str] = Counter()
    updates: list[tuple[int | None, str | None, int | None, int]] = []
    for media_id, lens_model in conn.execute("SELECT id, lens_model FROM media").fetchall():
        front = is_front_camera(lens_model)
        shot: str | None = None
        selfie: int | None = None
        if int(media_id) in kept and int(media_id) in scanned:
            shot = shot_type(fg_count[int(media_id)])
            selfie = int(is_selfie(front, fg_count[int(media_id)]))
            shots[shot] += 1
            shots["selfie"] += selfie
        updates.append((None if front is None else int(front), shot, selfie, int(media_id)))
    conn.executemany(
        "UPDATE media SET is_front_camera = ?, shot = ?, is_selfie = ? WHERE id = ?", updates
    )

    assigned = [pid for pid in person if pid is not None]
    present = set(assigned)
    return PeopleStats(
        people=len(present),
        named=sum(1 for pid in present if people_rows[pid]["name"]),
        photos_with_people=len({int(media[i]) for i, pid in enumerate(person) if pid}),
        foreground_faces=int((fg & in_kept).sum()),
        clustered=match.count("cluster"),
        attached=match.count("attached"),
        background=background,
        unassigned=sum(1 for i in candidates if person[i] is None),
        shots={kind: shots[kind] for kind in SHOT_KINDS},
        not_scanned=len(kept - scanned),
        joined=[
            tuple(display_name(people_rows[p]["number"], people_rows[p]["name"]) for p in group)
            for group in carry.joined
        ],
        refs_unmatched=sorted(set(refs) - set(matches.matched) - matches.settled),
        suggestions=suggest_merges(conn, settings.suggest_min_similarity),
    )


def _kept_images(conn: sqlite3.Connection) -> set[int]:
    """Images in all/: decodable, in some zip, and not a copy of another photo."""
    return {
        int(media_id)
        for (media_id,) in conn.execute(
            "SELECT id FROM media WHERE kind = 'image' AND status = 'ok' "
            "AND id IN (SELECT media_id FROM occurrences) "
            "AND id NOT IN (SELECT gm.media_id FROM group_members gm "
            "               JOIN media_groups g ON g.id = gm.group_id "
            "               WHERE g.kind = 'copy' AND gm.is_keeper = 0)"
        )
    }


def _people(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    return {
        int(row[0]): {"number": int(row[1]), "name": row[2], "named_by": row[3]}
        for row in conn.execute("SELECT id, number, name, named_by FROM people")
    }


def _create_people(conn: sqlite3.Connection, count: int) -> list[int]:
    """Insert ``count`` unnamed people with fresh numbers; numbers are never reused."""
    (highest,) = conn.execute("SELECT COALESCE(MAX(number), 0) FROM people").fetchone()
    start = max(int(get_meta(conn, NEXT_NUMBER_KEY) or 1), int(highest) + 1)
    ids = []
    for offset in range(count):
        cursor = conn.execute("INSERT INTO people (number) VALUES (?)", (start + offset,))
        assert cursor.lastrowid is not None
        ids.append(cursor.lastrowid)
    set_meta(conn, NEXT_NUMBER_KEY, str(start + count))
    return ids


def _apply_ref_names(
    conn: sqlite3.Connection, people: dict[int, dict[str, Any]], matched: dict[str, int]
) -> None:
    for pid, row in people.items():  # first drop the names refs no longer give
        if row["named_by"] == "refs" and matched.get(str(row["name"])) != pid:
            conn.execute(
                "UPDATE people SET name = NULL, name_key = NULL, named_by = NULL WHERE id = ?",
                (pid,),
            )
    for name, pid in sorted(matched.items()):
        row = people[pid]
        if row["named_by"] == "refs" and row["name"] == name:
            continue
        conn.execute(
            "UPDATE people SET name = ?, name_key = ?, named_by = 'refs' WHERE id = ?",
            (name, name_key(name), pid),
        )
