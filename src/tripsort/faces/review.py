"""``review/people.html``: merge suggestions, a grid of sample face crops per person, and
faces that belong to nobody, to check the people folders before sharing them.

Crops are read back from the zips (the index keeps no pixels) and embedded in the page, so
the page is one file. Each person's samples run from the faces most like their average face
to the least alike, where wrong faces show up. A merge suggestion shows every face of the
person suggested, least like the named person first, so one stranger in a mostly right
cluster is seen before the merge. The page shows friends' faces: it stays on this computer,
and `tripsort clean` deletes it along with the index.
"""

import base64
import html
import io
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from tripsort.faces.names import display_name
from tripsort.faces.suggest import Suggestion, commands
from tripsort.faces.types import EMBEDDING_SIZE, FACE_MODEL_ID
from tripsort.ingest.decode import UnreadableImageError, decode_oriented
from tripsort.ingest.sources import ZipCache
from tripsort.progress import NullProgress, Progress
from tripsort.stats import count

REVIEW_DIR = "review"
PAGE_NAME = "people.html"
CROP_HEIGHT = 112
COMPARE_FACES = 8  # a named person's most typical faces, next to a suggestion
_PADDING = 0.25  # of the face size, on each side


@dataclass(frozen=True)
class _Face:
    id: int
    media_id: int
    sha256: str
    box: tuple[float, float, float, float]
    match: str | None
    similarity: float | None
    caption: str | None = None  # instead of the similarity


@dataclass
class _Section:
    title: str
    note: str
    rows: list[tuple[str, list[_Face]]]  # (label, faces); an empty label for a plain grid
    commands: list[str] = field(default_factory=list)
    level: int = 2  # the heading level


def write_review(
    conn: sqlite3.Connection,
    project_dir: Path,
    *,
    per_person: int = 24,
    unassigned: int = 120,
    progress: Progress | None = None,
    suggestions: list[Suggestion] | None = None,
    project: str | None = None,
) -> Path:
    progress = progress or NullProgress()
    sections = _suggestion_sections(conn, suggestions or [], project or str(project_dir))
    for person_id, number, name, named_by in conn.execute(
        "SELECT p.id, p.number, p.name, p.named_by FROM people p "
        "WHERE EXISTS (SELECT 1 FROM faces f WHERE f.person_id = p.id AND f.match IS NOT NULL) "
        "ORDER BY (SELECT COUNT(DISTINCT media_id) FROM faces f "
        "          WHERE f.person_id = p.id AND f.match IS NOT NULL) DESC, p.number"
    ).fetchall():
        faces = _person_faces(conn, int(person_id))
        photos = len({f.media_id for f in faces})
        how = {"refs": "named from refs/", "user": "named by you"}.get(named_by, "not named yet")
        title = display_name(int(number), name)
        note = f"{photos:,} photos, {len(faces):,} faces, {how}"
        sections.append(_Section(title, note, [("", _sample(faces, per_person))]))
    loose = _unassigned(conn, unassigned)
    if loose:
        sections.append(
            _Section(
                "Faces in nobody's folder",
                "foreground faces that matched no one: strangers, or friends to look for",
                [("", loose)],
            )
        )

    every_face = [f for s in sections for _, faces in s.rows for f in faces]
    crops = _crops(conn, every_face, progress)
    names = _photo_names(conn)
    page = _page(sections, crops, names)
    folder = project_dir / REVIEW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / PAGE_NAME
    tmp = folder / f".{PAGE_NAME}.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(path)
    return path


def _suggestion_sections(
    conn: sqlite3.Connection, suggestions: list[Suggestion], project: str
) -> list[_Section]:
    """For each suggestion: every face of the unnamed person, least like the (first) named
    person first, then each named person's most typical faces, and the commands."""
    if not suggestions:
        return []
    ids = {int(number): int(pid) for pid, number in conn.execute("SELECT id, number FROM people")}
    sections = [
        _Section(
            "Maybe the same person",
            "tripsort never merges people on its own. For each suggestion below, look at every "
            "face first: one stranger among them is reason enough to dismiss it.",
            [],
        )
    ]
    for s in suggestions:
        candidates = [(c, ids[c.number]) for c in s.candidates]
        centroids = [_centroid(conn, pid) for _, pid in candidates]
        if any(c is None for c in centroids):  # pragma: no cover - suggestions need both
            continue
        first = candidates[0][0].name
        faces = []
        for face, vector in _face_rows(conn, ids[s.number]):
            similarities = [float(vector @ c) for c in centroids if c is not None]
            caption = " / ".join(f"{x:.2f}" for x in similarities)
            faces.append(
                _Face(
                    face.id,
                    face.media_id,
                    face.sha256,
                    face.box,
                    face.match,
                    similarities[0],
                    caption,
                )
            )
        faces.sort(key=lambda f: (f.similarity or 0.0, f.sha256, f.id))
        names = " or ".join(c.name for c, _ in candidates)
        against = " / ".join(c.name for c, _ in candidates)
        similar = ", ".join(f"{c.name} {c.similarity:.2f}" for c, _ in candidates)
        rows = [
            (f"Every face of {s.person}, least like {first} first (similarity to {against})", faces)
        ]
        for c, pid in candidates:
            rows.append((f"{c.name}: most typical faces", _person_faces(conn, pid)[:COMPARE_FACES]))
        sections.append(
            _Section(
                f"{s.person} may be {names}",
                f"{count(s.photos, 'photo')}, {count(len(faces), 'face')}; similarity of the "
                f"average faces: {similar}",
                rows,
                [command for _, command in commands(s, project)],
                level=3,
            )
        )
    return sections


def _centroid(conn: sqlite3.Connection, person_id: int) -> npt.NDArray[np.float32] | None:
    row = conn.execute("SELECT centroid FROM people WHERE id = ?", (person_id,)).fetchone()
    if not row or not row[0]:
        return None
    return np.frombuffer(row[0], dtype=np.float32, count=EMBEDDING_SIZE)


def _face_rows(
    conn: sqlite3.Connection, person_id: int
) -> list[tuple[_Face, npt.NDArray[np.float32]]]:
    """Each face that belongs to the person now, with its embedding."""
    return [
        (
            _Face(int(face_id), int(media_id), str(sha), (x, y, w, h), match, None),
            np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE),
        )
        for face_id, media_id, sha, x, y, w, h, match, blob in conn.execute(
            "SELECT f.id, f.media_id, m.sha256, f.x, f.y, f.w, f.h, f.match, f.embedding "
            "FROM faces f JOIN media m ON m.id = f.media_id "
            "WHERE f.person_id = ? AND f.match IS NOT NULL ORDER BY m.sha256, f.face_index",
            (person_id,),
        )
    ]


def _person_faces(conn: sqlite3.Connection, person_id: int) -> list[_Face]:
    """The person's faces, most like their average face first."""
    centroid = _centroid(conn, person_id)
    faces = []
    for face, vector in _face_rows(conn, person_id):
        similarity = float(vector @ centroid) if centroid is not None else None
        faces.append(_Face(face.id, face.media_id, face.sha256, face.box, face.match, similarity))
    faces.sort(key=lambda f: -(f.similarity if f.similarity is not None else 0.0))
    return faces


def _sample(faces: list[_Face], count: int) -> list[_Face]:
    """Evenly spread over the faces, which are sorted from most to least alike: the least
    alike, where mistakes hide, are always included."""
    if len(faces) <= count:
        return faces
    picks = np.linspace(0, len(faces) - 1, count).round().astype(int)
    return [faces[i] for i in sorted(set(picks.tolist()))]


def _unassigned(conn: sqlite3.Connection, limit: int) -> list[_Face]:
    rows = conn.execute(
        "SELECT f.id, f.media_id, m.sha256, f.x, f.y, f.w, f.h FROM faces f "
        "JOIN media m ON m.id = f.media_id WHERE f.model_id = ? AND f.is_foreground = 1 "
        "AND f.match IS NULL AND f.cluster IS NOT NULL ORDER BY f.h DESC, m.sha256 LIMIT ?",
        (FACE_MODEL_ID, limit),
    ).fetchall()
    return [
        _Face(int(r[0]), int(r[1]), str(r[2]), (r[3], r[4], r[5], r[6]), None, None) for r in rows
    ]


def _crops(conn: sqlite3.Connection, faces: list[_Face], progress: Progress) -> dict[int, str]:
    """face id -> a JPEG data URI of the face, padded a little."""
    by_photo: defaultdict[int, list[_Face]] = defaultdict(list)
    for face in faces:
        by_photo[face.media_id].append(face)
    crops: dict[int, str] = {}
    progress.start("review", len(by_photo), "files")
    try:
        with ZipCache(conn) as zips:
            for media_id, wanted in sorted(by_photo.items()):
                data = zips.read(media_id, wanted[0].sha256)
                progress.advance(1)
                if data is None:
                    continue
                try:
                    pixels = decode_oriented(data)
                except UnreadableImageError:
                    continue
                image = Image.fromarray(pixels)
                for face in wanted:
                    crops[face.id] = _crop_uri(image, face.box)
    finally:
        progress.finish()
    return crops


def _crop_uri(image: Image.Image, box: tuple[float, float, float, float]) -> str:
    width, height = image.size
    x, y, w, h = box
    pad_w, pad_h = w * _PADDING, h * _PADDING
    left = max(0, int((x - pad_w) * width))
    top = max(0, int((y - pad_h) * height))
    right = min(width, max(left + 1, int((x + w + pad_w) * width)))
    bottom = min(height, max(top + 1, int((y + h + pad_h) * height)))
    crop = image.crop((left, top, right, bottom))
    scale = CROP_HEIGHT / crop.height
    crop = crop.resize((max(1, round(crop.width * scale)), CROP_HEIGHT), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _photo_names(conn: sqlite3.Connection) -> dict[int, str]:
    """media id -> how the user finds the photo: its all/ path, or its zip and name."""
    names: dict[int, str] = {}
    for media_id, label, member in conn.execute(
        "SELECT o.media_id, s.label, o.member_path FROM occurrences o "
        "JOIN sources s ON s.id = o.source_id ORDER BY s.label, o.member_path"
    ):
        names.setdefault(int(media_id), f"{label}: {member}")
    for media_id, relpath in conn.execute(
        "SELECT media_id, relpath FROM exported_files WHERE role = 'all'"
    ):
        names[int(media_id)] = str(relpath)
    return names


_STYLE = """
:root { --bg: #fafaf9; --fg: #1c1917; --muted: #78716c; --card: #ffffff; --line: #e7e5e4;
        --accent: #2563eb; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1c1917; --fg: #f5f5f4; --muted: #a8a29e; --card: #292524; --line: #44403c;
          --accent: #60a5fa; }
}
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg);
       font: 15px/1.45 system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 2px; }
h3 { font-size: 15px; margin: 24px 0 2px; }
p { margin: 0 0 8px; color: var(--muted); }
p.row { margin: 12px 0 0; font-size: 13px; }
pre { margin: 6px 0 0; padding: 8px 10px; background: var(--card); border: 1px solid var(--line);
      border-radius: 6px; overflow-x: auto; font-size: 12px; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
figure { margin: 0; background: var(--card); border: 1px solid var(--line); border-radius: 6px;
         padding: 4px; text-align: center; }
figure img { display: block; height: 112px; border-radius: 3px; }
figure.background { border-style: dashed; }
figure.attached { border-color: var(--accent); }
figcaption { font-size: 11px; color: var(--muted); }
.missing { width: 84px; height: 112px; display: grid; place-items: center; font-size: 11px; }
"""


def _page(sections: list[_Section], crops: dict[int, str], names: dict[int, str]) -> str:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>People review</title><style>{_STYLE}</style></head><body><main>",
        "<h1>People review</h1>",
        "<p>Each row runs from the faces most like the person to the least alike. Dashed: "
        "matched in the background. Blue: joined the person after clustering. Fix a person "
        "split in two with <code>tripsort people merge</code>, keep two people apart with "
        "<code>tripsort people dismiss</code>, and name people with "
        "<code>tripsort people rename</code>. This page shows faces: keep it private.</p>",
    ]
    for section in sections:
        h = f"h{section.level}"
        parts.append(f"<{h}>{html.escape(section.title)}</{h}><p>{html.escape(section.note)}</p>")
        if section.commands:
            lines = "\n".join(html.escape(c) for c in section.commands)
            parts.append(f"<pre><code>{lines}</code></pre>")
        for label, faces in section.rows:
            if label:
                parts.append(f"<p class='row'>{html.escape(label)}</p>")
            parts.append("<div class='grid'>")
            parts.extend(_figure(face, crops, names) for face in faces)
            parts.append("</div>")
    parts.append("</main></body></html>\n")
    return "".join(parts)


def _figure(face: _Face, crops: dict[int, str], names: dict[int, str]) -> str:
    css = html.escape(face.match or "")
    caption = face.caption
    if caption is None:
        caption = "" if face.similarity is None else f"{face.similarity:.2f}"
    tip = html.escape(names.get(face.media_id, ""), quote=True)
    uri = crops.get(face.id)
    image = (
        f"<img src='{uri}' alt='' loading='lazy'>"
        if uri
        else "<div class='missing'>zip missing</div>"
    )
    return (
        f"<figure class='{css}' title='{tip}'>{image}"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )
