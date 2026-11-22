"""The public Python API. The CLI is a thin layer that formats what these methods return.

from tripsort import Project

with Project.open("sorted/") as proj:          # creates sorted/.tripsort/index.db
    proj.ingest(["mine.zip", "sam.zip"])       # stage 1: decode each new file once
    proj.people.name_from_refs("refs/")        # refs/<name>.jpg
    proj.analyze()                             # dates, dedupe, people: queries only
    proj.people.rename("person_03", "alex")
    proj.people.suggestions()                  # unnamed people who may be someone named
    proj.people.merge("person_07", into="alex")
    proj.people.dismiss("person_09", "alex")   # not alex: stop suggesting it
    result = proj.export(hardlinks=True)       # all/, people/, shots/, manifest.json
    print(result.summary())
"""

import json
import os
import shutil
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from tripsort import __version__
from tripsort._bench import peak_rss_bytes
from tripsort.clip.bursts import BurstStats
from tripsort.clip.categorize import CategoryStats
from tripsort.clip.store import pending_clip_scans
from tripsort.clip.types import CLIP_MODEL_ID, ImageEncoder, TextEncoder
from tripsort.config import OTHER, Config, load_config
from tripsort.cull.analysis import CullStats
from tripsort.dedupe.copies import rebuild_copy_groups
from tripsort.errors import ModelError, UserError, ZipError
from tripsort.export.manifest import write_manifest
from tripsort.export.writer import ExportResult, export_files
from tripsort.faces.analysis import PeopleStats
from tripsort.faces.people import People, PersonInfo, RefsResult
from tripsort.faces.review import REVIEW_DIR
from tripsort.faces.store import pending_scans
from tripsort.faces.suggest import Suggestion, suggestion_lines
from tripsort.faces.types import FACE_MODEL_ID, FaceAnalyzer
from tripsort.index.db import (
    INDEX_DIR_NAME,
    get_meta,
    index_path,
    open_index,
    set_meta,
    transaction,
)
from tripsort.ingest.dates import resolve_dates
from tripsort.ingest.pipeline import IngestResult, ingest_zips
from tripsort.progress import Progress
from tripsort.stats import ProjectStats, count, human_bytes, project_stats

__all__ = [
    "AnalyzeResult",
    "ExportResult",
    "IngestResult",
    "People",
    "PeopleStats",
    "PersonInfo",
    "Project",
    "ProjectStats",
    "RefsResult",
    "Suggestion",
    "clean",
]

# bumped by rename, merge, and dismiss, so the next export re-analyzes
PEOPLE_EDITS_KEY = "people_edits"
SUMMARY_SUGGESTIONS = 5  # merge suggestions in a summary; `people list` shows them all


@dataclass
class AnalyzeResult:
    stats: ProjectStats
    dates: dict[str, int] = field(default_factory=dict)  # how many dates came from each source
    people: PeopleStats = field(default_factory=PeopleStats)
    seconds: float = 0.0
    peak_rss_bytes: int | None = None
    folder: Path | None = None  # the project folder
    categories: CategoryStats = field(default_factory=CategoryStats)
    bursts: BurstStats = field(default_factory=BurstStats)
    culling: CullStats = field(default_factory=CullStats)

    def summary(self, project: str | None = None) -> str:
        """The summary. ``project`` is how its commands name the project folder, such as the
        path the user typed (default: the folder's full path)."""
        if project is None:
            project = str(self.folder) if self.folder is not None else "PROJECT"
        s = self.stats
        lines = [
            f"{count(s.files_found, 'file')} in {count(s.sources, 'zip')}: {s.unique:,} unique, "
            f"{count(s.exact_duplicates, 'exact duplicate')}."
        ]
        if s.near_copies:
            lines.append(
                f"{count(s.near_copies, 'recompressed copy', 'recompressed copies')} of "
                f"{count(s.copy_groups, 'photo')}; the best version of each is kept."
            )
        lines.append(
            f"Keeping {count(s.kept, 'file')} ({human_bytes(s.bytes_kept)}), "
            f"{human_bytes(s.bytes_saved)} saved."
        )
        if s.not_analyzed:
            lines.append(
                f"{count(s.not_analyzed, 'file')} (videos, very large images) checked for "
                "exact duplicates only."
            )
        if s.unreadable:
            lines.append(f"{count(s.unreadable, 'file')} could not be decoded (see unreadable/).")
        if s.read_failures:
            lines.append(
                f"{count(s.read_failures, 'zip entry', 'zip entries')} could not be read "
                "(see the messages above and manifest.json)."
            )
        lines.extend(_category_lines(self.categories, self.bursts, self.culling))
        lines.extend(_people_lines(self.people, project))
        return "\n".join(lines)


def _category_lines(c: CategoryStats, b: BurstStats, q: CullStats) -> list[str]:
    lines = []
    if c.problem:
        lines.append(f"Categories were not scored: {c.problem}")
    elif c.scored:
        sorted_into = [f"{n:,} {label}" for label, n in c.photos.items() if label != OTHER and n]
        lines.append(
            f"Sorted {count(c.scored, 'photo')} by category: "
            f"{', '.join(sorted_into) if sorted_into else 'none fits one'}; "
            f"{c.photos.get(OTHER, 0):,} other."
        )
    if c.not_embedded:
        lines.append(
            f"{count(c.not_embedded, 'photo')} without a CLIP embedding yet "
            "(no category, no burst); `tripsort ingest` on this project embeds them."
        )
    if b.bursts:
        lines.append(
            f"Found {count(b.bursts, 'burst')} ({count(b.photos, 'photo')}); the best shot of "
            "each goes to the people, shot, and category folders, every shot stays in all/, "
            "and bursts/ shows them side by side."
        )
    if rejected := sum(q.rejects.values()):
        reasons = ", ".join(
            f"{n:,} {reason.replace('_', ' ')}" for reason, n in q.rejects.items() if n
        )
        lines.append(
            f"{count(rejected, 'photo')} flagged as rejects ({reasons}): they stay in all/ "
            "and go to rejects/, not the people, shot, or category folders."
        )
    if q.not_scored:
        lines.append(
            f"{count(q.not_scored, 'photo')} without sharpness and exposure scores yet; "
            "`tripsort ingest` on this project scores them."
        )
    return lines


def _people_lines(p: PeopleStats, project: str) -> list[str]:
    lines = []
    if p.people or p.foreground_faces:
        named = f" ({p.named:,} named)" if p.named else ""
        lines.append(
            f"Found {count(p.people, 'person', 'people')}{named} in "
            f"{count(p.photos_with_people, 'photo')}: {p.shots.get('solo', 0):,} solo, "
            f"{p.shots.get('group', 0):,} group, {count(p.shots.get('selfie', 0), 'selfie')}."
        )
    if p.unassigned:
        lines.append(
            f"{count(p.unassigned, 'foreground face')} matched nobody: strangers, or someone "
            "to merge (see `tripsort people review`)."
        )
    for group in p.joined:
        lines.append(
            f"The clustering put {' and '.join(group)} together; they were kept apart as you "
            "named them, but check `tripsort people review`."
        )
    if p.refs_unmatched:
        lines.append(f"No one looks enough like the refs/ photos of {', '.join(p.refs_unmatched)}.")
    lines.extend(suggestion_lines(p.suggestions, project, SUMMARY_SUGGESTIONS))
    if p.not_scanned:
        lines.append(
            f"{count(p.not_scanned, 'photo')} not scanned for faces yet; "
            "`tripsort ingest` on this project scans them."
        )
    return lines


class Project:
    """A project folder: the sorted output plus its index in ``.tripsort/``."""

    def __init__(self, folder: Path, conn: sqlite3.Connection, config: Config) -> None:
        self.folder = folder
        self.config = config
        self._conn = conn
        self._faces: FaceAnalyzer | None = None
        self._clip: ImageEncoder | None = None
        self._text: TextEncoder | None = None
        self.people = People(self)

    @classmethod
    def open(cls, folder: str | os.PathLike[str]) -> Self:
        """Open the project in ``folder``, creating the folder and its index if needed."""
        path = Path(folder).expanduser().resolve()
        if path.exists() and not path.is_dir():
            raise UserError(f"{path} exists and is not a folder")
        path.mkdir(parents=True, exist_ok=True)
        config = load_config(path)
        return cls(path, open_index(index_path(path)), config)

    @property
    def connection(self) -> sqlite3.Connection:
        """The index connection, for read-only queries (tests, evaluation)."""
        return self._conn

    def face_analyzer(self, progress: Progress | None = None) -> FaceAnalyzer:
        """The face analyzer, loaded once per project; the models download on first use.

        Raises ``ModelError`` when the models cannot be downloaded.
        """
        if self._faces is None:
            from tripsort.faces import analyzer  # OpenCV loads only when faces are needed

            self._faces = analyzer.load_face_analyzer(progress)
        return self._faces

    def clip_encoder(self, progress: Progress | None = None) -> ImageEncoder:
        """The CLIP image encoder, loaded once per project; the model downloads on first use.

        Raises ``ModelError`` when the model cannot be downloaded.
        """
        if self._clip is None:
            from tripsort.clip import encoder  # ONNX Runtime loads only when CLIP is needed

            self._clip = encoder.load_image_encoder(progress)
        return self._clip

    def ingest(
        self, zips: Iterable[str | os.PathLike[str]] = (), *, progress: Progress | None = None
    ) -> IngestResult:
        """Stage 1: read the zips and decode every file not seen before, finding faces and
        the CLIP embedding in the same pass. Photos indexed earlier without either are
        scanned too, so ``ingest()`` with no zips only catches up on them."""
        paths = [Path(z) for z in zips]
        for path in paths:  # before the models load (or download) for nothing
            if not path.is_file():
                raise ZipError(f"{path.resolve()}: no such file")
        face_problem: list[str] = []
        clip_problem: list[str] = []

        def load_faces() -> FaceAnalyzer | None:
            try:
                return self.face_analyzer(progress)
            except ModelError as exc:
                face_problem.append(str(exc))
                return None

        def load_clip() -> ImageEncoder | None:
            try:
                return self.clip_encoder(progress)
            except ModelError as exc:
                clip_problem.append(str(exc))
                return None

        with self._run("ingest") as run:
            result = ingest_zips(
                self._conn, paths, progress=progress, faces=load_faces, clip=load_clip
            )
            result.face_problem = face_problem[0] if face_problem else None
            result.faces_pending = pending_scans(
                self._conn, self._faces.model_id if self._faces else FACE_MODEL_ID
            )
            result.clip_problem = clip_problem[0] if clip_problem else None
            result.clip_pending = pending_clip_scans(
                self._conn, self._clip.model_id if self._clip else CLIP_MODEL_ID
            )
            run.items, run.bytes = result.files_read, result.bytes_read
        result.peak_rss_bytes = run.peak_rss_bytes
        return result

    def analyze(self) -> AnalyzeResult:
        """Resolve dates, find copies, score categories, sort faces into people and shot
        types, and group bursts. Reads the index only, never the zips; the text encoder
        loads only for prompts whose embeddings are neither shipped nor cached."""
        from tripsort.clip.bursts import analyze_bursts
        from tripsort.clip.categorize import analyze_categories
        from tripsort.cull.analysis import analyze_culling
        from tripsort.faces.analysis import analyze_people  # pulls in scikit-learn

        with self._run("analyze") as run:
            started = time.perf_counter()
            dates = resolve_dates(self._conn)
            rebuild_copy_groups(self._conn, self.config.thresholds)
            try:
                with transaction(self._conn):
                    # categories first: the shot types use their selfie hint
                    categories = analyze_categories(self._conn, self.config, self.text_encoder)
                    people = analyze_people(self._conn, self.config.people)
                    bursts = analyze_bursts(self._conn, self.config.thresholds)
                    culling = analyze_culling(self._conn, self.config)
            finally:
                self._text = None  # the text encoder is only needed for new prompts
            set_meta(self._conn, "analysis_state", self._analysis_state())
            stats = project_stats(self._conn)
            run.items = stats.unique
        return AnalyzeResult(
            stats,
            dates,
            people,
            time.perf_counter() - started,
            run.peak_rss_bytes,
            self.folder,
            categories,
            bursts,
            culling,
        )

    def text_encoder(self, progress: Progress | None = None) -> "TextEncoder":
        """The CLIP text encoder, for prompts tripsort does not ship; downloads on first use.

        Raises ``ModelError`` when it cannot be downloaded.
        """
        if self._text is None:
            from tripsort.clip import encoder

            self._text = encoder.load_text_encoder(progress)
        return self._text

    def export(
        self,
        *,
        hardlinks: bool = True,
        dry_run: bool = False,
        export_duplicates: bool = False,
        progress: Progress | None = None,
    ) -> ExportResult:
        """Stage 6: write all/, unreadable/, optionally duplicates/, and manifest.json.

        Runs ``analyze`` first when the index changed since the last analysis.
        """
        if self.analysis_is_stale():
            self.analyze()
        with self._run("export") as run:
            result = export_files(
                self._conn,
                self.folder,
                hardlinks=hardlinks,
                dry_run=dry_run,
                export_duplicates=export_duplicates,
                progress=progress,
            )
            if not dry_run:
                result.manifest = write_manifest(
                    self._conn, self.folder, self.config, project_stats(self._conn)
                )
            run.items, run.bytes = result.written, result.bytes_written
        result.peak_rss_bytes = run.peak_rss_bytes
        return result

    def stats(self) -> ProjectStats:
        return project_stats(self._conn)

    def analysis_is_stale(self) -> bool:
        return get_meta(self._conn, "analysis_state") != self._analysis_state()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _analysis_state(self) -> str:
        counts = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM media), (SELECT COALESCE(MAX(id), 0) FROM media), "
            "(SELECT COUNT(*) FROM occurrences), (SELECT COUNT(*) FROM sources), "
            "(SELECT COUNT(*) FROM face_scans), (SELECT COALESCE(MAX(id), 0) FROM faces), "
            "(SELECT COUNT(*) FROM clip_scans), "
            "(SELECT COALESCE(GROUP_CONCAT(relpath || ':' || sha256, '|'), '') "
            " FROM (SELECT relpath, sha256 FROM ref_images ORDER BY relpath))"
        ).fetchone()
        edits = get_meta(self._conn, PEOPLE_EDITS_KEY) or "0"
        return json.dumps({"config": self.config.hash, "counts": list(counts), "edits": edits})

    def _people_edited(self) -> None:
        """Called after a rename, merge, or dismiss: the next export analyzes again (index
        only), so background faces and names follow the edit."""
        edits = int(get_meta(self._conn, PEOPLE_EDITS_KEY) or "0")
        set_meta(self._conn, PEOPLE_EDITS_KEY, str(edits + 1))

    @contextmanager
    def _run(self, stage: str) -> Iterator["_RunRecord"]:
        record = _RunRecord()
        cursor = self._conn.execute(
            "INSERT INTO runs (started_at, tool_version, config_hash, stage) VALUES (?, ?, ?, ?)",
            (_now(), __version__, self.config.hash, stage),
        )
        started = time.perf_counter()
        yield record
        record.peak_rss_bytes = peak_rss_bytes()
        self._conn.execute(
            "UPDATE runs SET finished_at = ?, items = ?, bytes = ?, seconds = ?, "
            "peak_rss_bytes = ? WHERE id = ?",
            (
                _now(),
                record.items,
                record.bytes,
                time.perf_counter() - started,
                record.peak_rss_bytes,
                cursor.lastrowid,
            ),
        )


@dataclass
class _RunRecord:
    items: int = 0
    bytes: int = 0
    peak_rss_bytes: int | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def clean(folder: str | os.PathLike[str]) -> bool:
    """Delete the project's index (``.tripsort/``: hashes, faces, embeddings, names) and the
    review pages (``review/``, which show face crops). Exported folders are kept.

    Returns False when there was nothing to delete.
    """
    project = Path(folder).expanduser().resolve()
    deleted = False
    for name in (INDEX_DIR_NAME, REVIEW_DIR):
        target = project / name
        if target.exists():
            shutil.rmtree(target)
            deleted = True
    return deleted
