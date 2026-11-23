"""People in videos. Reads and writes the index only.

Faces in sampled video frames are not clustered: each counted one (detection score and size as
the people settings count them) is matched to the person, found in photos, whose average face
is most like it, if at least ``attach_min_similarity`` alike. A person is in a video when
matched in at least ``person_min_frames`` of its frames; those faces then carry the person (as
attached faces, like photo faces that join the most similar person), so the video goes into
that person's people/ folder. Live Photo motion clips are never sampled.

Matching is off unless ``[video] match_people`` is true: on a labeled trip it found too few of
the people in videos, and not reliably enough, for people/ folders. Off, videos get no people
(they still go into all/ and categories/; faces still decide their `other` category).
"""

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from tripsort.config import Config
from tripsort.faces.types import EMBEDDING_SIZE, FACE_MODEL_ID


@dataclass(frozen=True)
class VideoPeopleStats:
    videos: int = 0  # videos with at least one person in them
    appearances: int = 0  # (video, person) pairs


def analyze_video_people(conn: sqlite3.Connection, config: Config) -> VideoPeopleStats:
    """Match every sampled frame's faces to people. Call inside a transaction, after the
    people step has stored each person's average face."""
    people, video = config.people, config.video
    conn.execute("UPDATE faces SET person_id = NULL, match = NULL WHERE frame_ts >= 0")
    if not video.match_people:
        return VideoPeopleStats()  # and earlier matches are gone, so no video keeps a person
    centroids = [
        (int(pid), np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE))
        for pid, blob in conn.execute("SELECT id, centroid FROM people WHERE centroid IS NOT NULL")
    ]
    if not centroids:
        return VideoPeopleStats()
    ids = [pid for pid, _ in centroids]
    matrix = np.stack([vector for _, vector in centroids])
    hits: defaultdict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for face_id, media_id, frame_ts, blob in conn.execute(
        "SELECT id, media_id, frame_ts, embedding FROM faces WHERE model_id = ? AND frame_ts >= 0 "
        "AND score >= ? AND h >= ? AND media_id NOT IN (SELECT motion_id FROM live_photos)",
        (FACE_MODEL_ID, people.min_face_score, people.foreground_min_height),
    ):
        similarity = matrix @ np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE)
        best = int(np.argmax(similarity))
        if float(similarity[best]) >= people.attach_min_similarity:
            hits[(int(media_id), ids[best])].append((float(frame_ts), int(face_id)))
    found = {
        key: faces
        for key, faces in hits.items()
        if len({frame for frame, _ in faces}) >= video.person_min_frames
    }
    conn.executemany(
        "UPDATE faces SET person_id = ?, match = 'attached' WHERE id = ?",
        [(pid, face_id) for (_, pid), faces in found.items() for _, face_id in faces],
    )
    return VideoPeopleStats(videos=len({media for media, _ in found}), appearances=len(found))
