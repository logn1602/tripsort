"""Is any ref photo a copy of a photo in the labeled set?

A ref photo that is one of the labeled photos names that photo trivially and makes the scores
look better than they are. So each ref photo is checked three ways against the labeled set
in the index:

* the same file: its SHA-256 is a labeled photo's;
* a near copy: its pHash is within tripsort's loosest near-copy distance of a labeled
  photo's (``thresholds.phash_max_distance``), which catches resized and re-saved copies;
* the same face: its face (the largest, the one naming uses) is at least ``SAME_FACE``
  similar to a face found in the labeled set, which also catches crops and edits a pHash
  cannot see, and burst neighbors.

Only counts and positions leave this module. A flagged photo is given by its name's place in
alphabetical order (ignoring case) and its place among that name's photos sorted by path, so
the labeler can find it and the output names nobody.
"""

import hashlib
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tripsort.faces.naming import ref_files
from tripsort.faces.types import EMBEDDING_SIZE
from tripsort.ingest.decode import UnreadableImageError, analyze_image
from tripsort.ingest.hashing import hamming, phash_to_int

# Nearly the same face: the same shot cropped or edited, or a burst neighbor. Different photos
# of one person stayed at 0.81 or below in the labeled set.
SAME_FACE = 0.9


def ref_copies(
    conn: sqlite3.Connection, folder: Path, model_id: str, max_distance: int
) -> dict[str, Any]:
    """Check every photo in the refs ``folder`` against the labeled set in ``conn``, an index
    that has stored those refs (``People.name_from_refs``)."""
    labeled_sha = {str(sha) for (sha,) in conn.execute("SELECT sha256 FROM media")}
    phashes = [
        phash_to_int(str(p))
        for (p,) in conn.execute("SELECT phash FROM media WHERE phash IS NOT NULL")
    ]
    blobs = [
        bytes(blob)
        for (blob,) in conn.execute("SELECT embedding FROM faces WHERE model_id = ?", (model_id,))
    ]
    faces = (
        np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), EMBEDDING_SIZE)
        if blobs
        else np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    )
    ref_faces = {
        (str(relpath), str(sha)): bytes(blob)
        for relpath, sha, blob in conn.execute(
            "SELECT relpath, sha256, embedding FROM ref_images "
            "WHERE model_id = ? AND status = 'ok'",
            (model_id,),
        )
    }
    files, _ = ref_files(folder)  # sorted by path
    names = sorted({f.name for f in files}, key=lambda n: (n.casefold(), n))
    position = {name: i for i, name in enumerate(names, 1)}
    photos_of = Counter(f.name for f in files)
    seen: Counter[str] = Counter()
    found: Counter[str] = Counter()
    distances: list[int] = []
    similarities: list[float] = []
    flagged = []
    for ref in files:
        seen[ref.name] += 1
        try:
            data = ref.path.read_bytes()
        except OSError:  # naming reported it already
            continue
        sha = hashlib.sha256(data).hexdigest()
        why = []
        if sha in labeled_sha:
            why.append("same file")
        try:
            phash = analyze_image(data).phash
        except UnreadableImageError:
            phash = None
        if phash is not None and phashes:
            distance = min(hamming(phash_to_int(phash), other) for other in phashes)
            distances.append(distance)
            if distance <= max_distance:
                why.append("near copy")
        blob = ref_faces.get((ref.relpath, sha))
        if blob is not None and len(faces):
            vector = np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE)
            similarity = float(np.max(faces @ vector))
            similarities.append(similarity)
            if similarity >= SAME_FACE:
                why.append("same face")
        found.update(why)
        if why:
            flagged.append(
                {
                    "name_position": position[ref.name],
                    "photo_position": seen[ref.name],
                    "photos_for_name": photos_of[ref.name],
                    "why": why,
                }
            )
    return {
        "ref_photos": len(files),
        "same_file": found["same file"],
        "near_copy": found["near copy"],
        "same_face": found["same face"],
        "closest_phash_distance": min(distances) if distances else None,
        "highest_face_similarity": round(max(similarities), 3) if similarities else None,
        "flagged": flagged,
    }
