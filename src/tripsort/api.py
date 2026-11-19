"""The public Python API. The CLI is a thin layer that formats what these methods return.

from tripsort import Project

with Project.open("sorted/") as proj:          # creates sorted/.tripsort/index.db
    proj.ingest(["mine.zip", "sam.zip"])       # stage 1: decode each new file once
    proj.analyze()                             # dates and dedupe: queries only
    result = proj.export(hardlinks=True)       # all/ + manifest.json
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
from tripsort.config import Config, load_config
from tripsort.dedupe.copies import rebuild_copy_groups
from tripsort.errors import UserError
from tripsort.export.manifest import write_manifest
from tripsort.export.writer import ExportResult, export_files
from tripsort.index.db import INDEX_DIR_NAME, get_meta, index_path, open_index, set_meta
from tripsort.ingest.dates import resolve_dates
from tripsort.ingest.pipeline import IngestResult, ingest_zips
from tripsort.progress import Progress
from tripsort.stats import ProjectStats, count, human_bytes, project_stats

__all__ = [
    "AnalyzeResult",
    "ExportResult",
    "IngestResult",
    "Project",
    "ProjectStats",
    "clean",
]


@dataclass
class AnalyzeResult:
    stats: ProjectStats
    dates: dict[str, int] = field(default_factory=dict)  # how many dates came from each source
    seconds: float = 0.0
    peak_rss_bytes: int | None = None

    def summary(self) -> str:
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
        return "\n".join(lines)


class Project:
    """A project folder: the sorted output plus its index in ``.tripsort/``."""

    def __init__(self, folder: Path, conn: sqlite3.Connection, config: Config) -> None:
        self.folder = folder
        self.config = config
        self._conn = conn

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

    def ingest(
        self, zips: Iterable[str | os.PathLike[str]], *, progress: Progress | None = None
    ) -> IngestResult:
        """Stage 1: read the zips and decode every file not seen before."""
        with self._run("ingest") as run:
            result = ingest_zips(self._conn, [Path(z) for z in zips], progress=progress)
            run.items, run.bytes = result.files_read, result.bytes_read
        result.peak_rss_bytes = run.peak_rss_bytes
        return result

    def analyze(self) -> AnalyzeResult:
        """Resolve dates and find copies. Reads the index only, never the zips."""
        with self._run("analyze") as run:
            started = time.perf_counter()
            dates = resolve_dates(self._conn)
            rebuild_copy_groups(self._conn, self.config.thresholds)
            set_meta(self._conn, "analysis_state", self._analysis_state())
            stats = project_stats(self._conn)
            run.items = stats.unique
        return AnalyzeResult(stats, dates, time.perf_counter() - started, run.peak_rss_bytes)

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
            "(SELECT COUNT(*) FROM occurrences), (SELECT COUNT(*) FROM sources)"
        ).fetchone()
        return json.dumps({"config": self.config.hash, "counts": list(counts)})

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
    """Delete the project's index (``.tripsort/``). Exported folders are kept.

    Returns False when there was no index to delete.
    """
    target = Path(folder).expanduser().resolve() / INDEX_DIR_NAME
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True
