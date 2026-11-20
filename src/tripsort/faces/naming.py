"""Name people from a refs/ folder: ``refs/<name>.jpg``, or ``refs/<name>/*.jpg``.

Each photo is decoded the way ingest decodes, and its largest face is embedded once (kept in
the index, keyed by the file's SHA-256). A name's reference is the average of its photos'
faces. References name the most similar people, one name per person and one person per name,
taking the most similar pairs first, and only pairs at least ``ref_min_similarity`` similar.
A person the user renamed keeps that name: a reference whose best match is such a person
names nobody else.
"""

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tripsort.errors import PeopleError, UserError
from tripsort.faces.names import check_name, name_key
from tripsort.faces.types import EMBEDDING_SIZE, FaceAnalyzer, normalized
from tripsort.index.db import transaction
from tripsort.ingest.decode import UnreadableImageError, analyze_image
from tripsort.ingest.zip_reader import IMAGE_EXTENSIONS, JUNK_NAMES

Vector = npt.NDArray[np.float32]
SECOND_FACE_RATIO = 0.5  # a second face at least this tall (relative) makes a ref ambiguous
_CACHED_PROBLEMS = {"no_face": "no face found", "unreadable": "could not decode it"}


@dataclass(frozen=True)
class RefFile:
    name: str
    relpath: str  # '/'-separated, relative to the refs folder
    path: Path


@dataclass
class RefsReport:
    """What was found in a refs folder. Names and paths are the user's own, for the CLI."""

    names: dict[str, int] = field(default_factory=dict)  # name -> photos with a usable face
    problems: list[str] = field(default_factory=list)


def ref_files(folder: Path) -> tuple[list[RefFile], list[str]]:
    """The photos in ``folder`` with the name each one stands for, plus problems."""
    if not folder.is_dir():
        raise UserError(f"{folder}: no such folder")
    found: list[RefFile] = []
    problems: list[str] = []
    spelling: dict[str, str] = {}  # name key -> the first spelling seen ('Sam.jpg', 'sam/')
    # Sorted by the exact relative path: Windows paths compare without case, which would
    # make the first spelling depend on the OS.
    for path in sorted(folder.rglob("*"), key=lambda p: p.relative_to(folder).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts) or path.name.lower() in JUNK_NAMES:
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        raw_name = rel.stem if len(rel.parts) == 1 else rel.parts[0]
        try:
            name = check_name(raw_name)
        except PeopleError as exc:
            problems.append(f"{rel.as_posix()}: {exc}")
            continue
        name = spelling.setdefault(name_key(name), name)
        found.append(RefFile(name, rel.as_posix(), path))
    return found, problems


def store_refs(conn: sqlite3.Connection, folder: Path, analyzer: FaceAnalyzer) -> RefsReport:
    """Embed the photos in ``folder`` (reusing earlier embeddings of unchanged files) and make
    them the project's references, replacing any earlier set."""
    files, problems = ref_files(folder)
    report = RefsReport(problems=problems)
    known = {
        (str(relpath), str(sha)): (str(status), int(faces), embedding)
        for relpath, sha, status, faces, embedding in conn.execute(
            "SELECT relpath, sha256, status, faces, embedding FROM ref_images WHERE model_id = ?",
            (analyzer.model_id,),
        )
    }
    rows: list[tuple[object, ...]] = []
    for ref in files:
        try:
            data = ref.path.read_bytes()
        except OSError as exc:
            report.problems.append(f"{ref.relpath}: could not be read ({exc.strerror or exc})")
            continue
        sha = hashlib.sha256(data).hexdigest()
        cached = known.get((ref.relpath, sha))
        if cached is not None:
            status, faces, blob = cached
            problem = _CACHED_PROBLEMS.get(status)
        else:
            status, faces, blob, problem = _embed_ref(data, analyzer)
        if problem:
            report.problems.append(f"{ref.relpath}: {problem}")
        report.names.setdefault(ref.name, 0)
        if status == "ok":
            report.names[ref.name] += 1
        rows.append((ref.relpath, analyzer.model_id, ref.name, sha, status, faces, blob))
    with transaction(conn):
        conn.execute("DELETE FROM ref_images")
        conn.executemany(
            "INSERT INTO ref_images (relpath, model_id, name, sha256, status, faces, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    for name, usable in sorted(report.names.items()):
        if usable == 0:
            report.problems.append(f"{name}: no usable photo, so this name is not used")
    return report


def _embed_ref(data: bytes, analyzer: FaceAnalyzer) -> tuple[str, int, bytes | None, str | None]:
    try:
        scan = analyze_image(data, faces=analyzer).faces
    except UnreadableImageError as exc:
        return "unreadable", 0, None, f"could not decode it ({exc})"
    if scan is None or scan.status != "ok":
        detail = scan.detail if scan is not None else None
        return "unreadable", 0, None, f"could not scan it for faces ({detail})"
    if not scan.faces:
        return "no_face", 0, None, "no face found"
    largest = scan.faces[0]
    problem = None
    if len(scan.faces) > 1 and scan.faces[1].h >= SECOND_FACE_RATIO * largest.h:
        problem = "more than one face; the largest was used"
    return "ok", len(scan.faces), largest.embedding.astype(np.float32).tobytes(), problem


def load_refs(conn: sqlite3.Connection, model_id: str) -> dict[str, Vector]:
    """Each name's reference: the normalized mean of its photos' faces."""
    vectors: dict[str, list[Vector]] = {}
    for name, blob in conn.execute(
        "SELECT name, embedding FROM ref_images WHERE model_id = ? AND status = 'ok' "
        "ORDER BY name, relpath",
        (model_id,),
    ):
        vectors.setdefault(str(name), []).append(
            np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE)
        )
    return {name: normalized(np.mean(vs, axis=0)) for name, vs in vectors.items()}


@dataclass
class RefMatches:
    matched: dict[str, int] = field(default_factory=dict)  # name -> person it now names
    # names that need no person: their best match is a person the user named, or the user
    # gave the same name to someone
    settled: set[str] = field(default_factory=set)


def match_refs(
    refs: dict[str, Vector],
    people: dict[int, Vector],
    user_named: dict[int, str],
    min_similarity: float,
) -> RefMatches:
    """Pair reference names with people. ``people`` maps person ids to their average face;
    ``user_named`` holds the people the user named (id -> name), which refs never rename."""
    result = RefMatches()
    taken = {name_key(name) for name in user_named.values()}
    result.settled = {name for name in refs if name_key(name) in taken}
    candidates = [
        (float(vector @ centroid), name, person)
        for name, vector in refs.items()
        if name not in result.settled
        for person, centroid in people.items()
    ]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_people: set[int] = set()
    for similarity, name, person in candidates:
        if similarity < min_similarity:
            break
        if name in result.matched or name in result.settled or person in used_people:
            continue
        used_people.add(person)
        if person in user_named:  # a user-named best match spends the reference
            result.settled.add(name)
        else:
            result.matched[name] = person
    return result
