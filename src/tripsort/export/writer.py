"""Write the export folders incrementally and safely.

* ``all/`` holds each kept file once, extracted from its zip. ``unreadable/`` holds files
  whose bytes could be read but not decoded. With ``export_duplicates``, ``duplicates/<keeper>/``
  shows each keeper (a hardlink into ``all/``) next to the copies that were left out.
* ``people/<name>/`` and ``shots/{solo,group,selfie}/`` hold hardlinks into ``all/``, so a
  group photo in three people's folders takes disk space once. A renamed or merged person's
  links move to the new folder.
* What an export writes is recorded in the index (``exported_files``). The next export
  writes only what is missing, removes files it wrote earlier that are no longer wanted
  (only if they are unchanged since), leaves files edited since the last export alone, and
  never overwrites or deletes a file it did not write.
* Files are written under a temporary name and renamed into place, with their SHA-256
  checked on the way, so an interrupted export never leaves a partial file under a final name.
"""

import hashlib
import os
import shutil
import sqlite3
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from tripsort.errors import ExportError
from tripsort.export.naming import export_name, fold_key, numbered, person_folder, split_extension
from tripsort.faces.names import display_name
from tripsort.index.db import transaction
from tripsort.ingest.sources import ZipCache
from tripsort.progress import NullProgress, Progress
from tripsort.stats import count, human_bytes

ALL_DIR = "all"
UNREADABLE_DIR = "unreadable"
DUPLICATES_DIR = "duplicates"
PEOPLE_DIR = "people"
SHOTS_DIR = "shots"
SHOT_FOLDERS = ("solo", "group", "selfie")
TMP_SUFFIX = ".tripsort-tmp"
_MANAGED_DIRS = (ALL_DIR, UNREADABLE_DIR, DUPLICATES_DIR, PEOPLE_DIR, SHOTS_DIR)
_LINK_ROLES = frozenset({"dup_keeper", "person", "shot"})  # hardlinks into all/
_SPACE_MARGIN = 64 * 1024 * 1024
_COMMIT_EVERY = 50


@dataclass(frozen=True)
class _Media:
    id: int
    sha256: str
    bytes: int
    status: str
    taken_at: str | None


@dataclass(frozen=True)
class _Occurrence:
    source_id: int
    member_index: int
    member_path: str
    label: str


@dataclass(frozen=True)
class _Record:
    relpath: str
    fold_key: str
    media_id: int
    role: str
    method: str
    bytes: int
    mtime_ns: int


@dataclass
class _Wanted:
    media: _Media
    role: str  # 'all', 'unreadable', 'dup_keeper', 'dup_copy', 'person', 'shot'
    folder: str
    name: str
    link_to: str | None = None  # for the link roles: the all/ file to link
    relpath: str = ""
    action: str = ""  # 'write', 'unchanged', 'adopt', 'edited'


@dataclass(frozen=True)
class ExportProblem:
    sha256: str
    where: str  # 'label: member path'
    message: str


@dataclass
class ExportResult:
    dry_run: bool
    planned: int  # files the export folders hold when this export is done
    written: int
    unchanged: int
    removed: int
    linked: int
    link_fallbacks: int  # copies made because the drive does not support hardlinks
    bytes_written: int
    kept: int  # files in all/
    unreadable: int  # files in unreadable/
    near_copies: int  # copies left out of all/
    duplicates_exported: bool
    left_alone: int  # files edited after an earlier export, which were not touched
    problems: list[ExportProblem]
    seconds: float
    people_folders: int = 0
    shots: dict[str, int] = field(default_factory=dict)  # 'solo', 'group', 'selfie' -> files
    manifest: Path | None = None
    peak_rss_bytes: int | None = None

    def summary(self) -> str:
        verb = "Would write" if self.dry_run else "Wrote"
        lines = [
            f"{verb} {count(self.written, 'file')} ({human_bytes(self.bytes_written)}), "
            f"{self.unchanged:,} already up to date, "
            f"{self.removed:,} {'to remove' if self.dry_run else 'removed'}.",
            f"all/ holds {count(self.kept, 'file')}"
            + (f"; unreadable/ holds {self.unreadable:,} for review" if self.unreadable else "")
            + ".",
        ]
        if self.people_folders or any(self.shots.values()):
            shots = ", ".join(f"{self.shots.get(s, 0):,} {s}" for s in SHOT_FOLDERS)
            lines.append(
                f"people/ holds {count(self.people_folders, 'folder')}; shots/ holds {shots}."
            )
        if self.near_copies:
            where = (
                "duplicates/ shows each one next to the file that was kept"
                if self.duplicates_exported
                else "export with --export-duplicates to review them in duplicates/"
            )
            left_out = count(self.near_copies, "near-copy", "near-copies")
            verb_was = "was" if self.near_copies == 1 else "were"
            lines.append(f"{left_out} {verb_was} left out of all/; {where}.")
        if self.link_fallbacks:
            lines.append(
                "This drive does not support hardlinks, so people/, shots/, and duplicates/ "
                "hold copies."
            )
        if self.left_alone:
            lines.append(
                f"{count(self.left_alone, 'file')} changed after the last export "
                "and left as they are."
            )
        if self.problems:
            lines.append(f"{count(len(self.problems), 'file')} could not be exported:")
            lines.extend(f"  {p.where}: {p.message}" for p in self.problems)
        return "\n".join(lines)


def export_files(
    conn: sqlite3.Connection,
    project_dir: Path,
    *,
    hardlinks: bool = True,
    dry_run: bool = False,
    export_duplicates: bool = False,
    progress: Progress | None = None,
) -> ExportResult:
    started = time.perf_counter()
    progress = progress or NullProgress()
    media = _load_media(conn)
    occurrences = _load_occurrences(conn)
    copies_by_keeper = _load_copy_groups(conn)
    copy_ids = {c for copies in copies_by_keeper.values() for c in copies}
    records = _load_records(conn)
    planner = _Planner(project_dir, records)

    by_date = sorted(
        (m for m in media.values() if m.id in occurrences),
        key=lambda m: (m.taken_at or "", m.sha256),
    )
    all_relpath: dict[int, str] = {}
    for m in by_date:
        name = _name_for(m, occurrences[m.id])
        if m.status == "unreadable":
            planner.want(_Wanted(m, "unreadable", UNREADABLE_DIR, name))
        elif m.id not in copy_ids:
            all_relpath[m.id] = planner.want(_Wanted(m, "all", ALL_DIR, name)).relpath
    if export_duplicates:
        for keeper_id in sorted(copies_by_keeper, key=lambda k: all_relpath.get(k, "")):
            keeper_rel = all_relpath.get(keeper_id)
            if keeper_rel is None:  # pragma: no cover - keepers always have an all/ entry
                continue
            keeper_name = PurePosixPath(keeper_rel).name
            folder = f"{DUPLICATES_DIR}/{split_extension(keeper_name)[0]}"
            keeper = _Wanted(media[keeper_id], "dup_keeper", folder, keeper_name, keeper_rel)
            planner.want(keeper)
            for copy_id in copies_by_keeper[keeper_id]:
                copy = media[copy_id]
                planner.want(
                    _Wanted(copy, "dup_copy", folder, _name_for(copy, occurrences[copy_id]))
                )
    people = _load_people_folders(conn)
    shots = _load_shot_folders(conn)
    for m in by_date:
        rel = all_relpath.get(m.id)
        if rel is None:
            continue
        name = PurePosixPath(rel).name
        for person in people.get(m.id, ()):
            planner.want(_Wanted(m, "person", f"{PEOPLE_DIR}/{person}", name, rel))
        for shot in shots.get(m.id, ()):
            planner.want(_Wanted(m, "shot", f"{SHOTS_DIR}/{shot}", name, rel))

    wanted = planner.wanted
    stale = [record for relpath, record in records.items() if relpath not in planner.claimed]
    writes = [w for w in wanted if w.action == "write"]
    needed = sum(w.media.bytes for w in writes if w.role not in _LINK_ROLES or not hardlinks)
    result = ExportResult(
        dry_run=dry_run,
        planned=len(wanted),
        written=len(writes),
        unchanged=sum(1 for w in wanted if w.action in {"unchanged", "adopt"}),
        removed=len(stale),
        linked=0,
        link_fallbacks=0,
        bytes_written=needed,
        kept=sum(1 for w in wanted if w.role == "all"),
        unreadable=sum(1 for w in wanted if w.role == "unreadable"),
        near_copies=len(copy_ids),
        duplicates_exported=export_duplicates,
        left_alone=sum(1 for w in wanted if w.action == "edited"),
        problems=[],
        seconds=0.0,
        people_folders=len({w.folder for w in wanted if w.role == "person"}),
        shots={
            shot: sum(1 for w in wanted if w.role == "shot" and w.folder.endswith(f"/{shot}"))
            for shot in SHOT_FOLDERS
        },
    )
    if dry_run:
        result.seconds = time.perf_counter() - started
        return result

    free = shutil.disk_usage(project_dir).free
    if needed > free - _SPACE_MARGIN:
        raise ExportError(
            f"the export needs {human_bytes(needed)} but only {human_bytes(free)} is free "
            f"on the drive holding {project_dir}"
        )

    _remove_temp_files(project_dir)
    removed, kept_stale = _remove_stale(conn, project_dir, stale)
    result.removed = removed
    result.left_alone += kept_stale
    with transaction(conn):
        for w in wanted:
            if w.action == "adopt":
                _save_record(conn, _record_for(project_dir, w, "extract"))

    writer = _Writer(conn, project_dir, hardlinks, occurrences)
    progress.start("export", len(writes), "files")
    extracts = [w for w in writes if w.role not in _LINK_ROLES]
    extracts.sort(key=lambda w: _source_order(occurrences[w.media.id]))
    links = [w for w in writes if w.role in _LINK_ROLES]
    try:
        with ZipCache(conn) as zips:
            for w in extracts:
                writer.extract(zips, w)
                progress.advance(1)
        for w in links:
            writer.link(w)
            progress.advance(1)
    finally:
        writer.flush()
        progress.finish()
    result.written = writer.written
    result.bytes_written = writer.bytes_written
    result.linked = writer.linked
    result.link_fallbacks = writer.link_fallbacks
    result.problems = writer.problems
    result.seconds = time.perf_counter() - started
    return result


def _name_for(media: _Media, occurrences: list[_Occurrence]) -> str:
    first = occurrences[0]
    return export_name(media.taken_at, first.label, first.member_path)


def _source_order(occurrences: list[_Occurrence]) -> tuple[int, int]:
    first = occurrences[0]
    return first.source_id, first.member_index


class _Planner:
    """Assigns each wanted file a path, reusing the path an earlier export gave it."""

    def __init__(self, project_dir: Path, records: dict[str, _Record]) -> None:
        self.project_dir = project_dir
        # Keyed by folder too: one photo can sit in several people folders.
        self.owners = {
            (r.media_id, r.role, PurePosixPath(r.relpath).parent.as_posix()): r
            for r in records.values()
        }
        self.by_key = {r.fold_key: r for r in records.values()}
        self.taken = set(self.by_key)
        self.claimed: set[str] = set()
        self.wanted: list[_Wanted] = []

    def want(self, w: _Wanted) -> _Wanted:
        previous = self.owners.get((w.media.id, w.role, w.folder))
        if previous is not None and previous.relpath not in self.claimed:
            w.relpath = previous.relpath
            w.action = _disk_state(self.project_dir, previous)
        else:
            name, n = w.name, 0
            while True:
                relpath = f"{w.folder}/{name}"
                key = fold_key(relpath)
                if self._recased(key, relpath, w):
                    w.action = "write"  # the old spelling is removed before writes start
                    break
                if key not in self.taken:
                    path = _abs(self.project_dir, relpath)
                    if not os.path.lexists(path):
                        w.action = "write"
                        break
                    if _same_content(path, w.media):  # e.g. written before `tripsort clean`
                        w.action = "adopt"
                        break
                n += 1
                name = numbered(w.name, n)
            self.taken.add(key)
            w.relpath = relpath
        self.claimed.add(w.relpath)
        self.wanted.append(w)
        return w

    def _recased(self, key: str, relpath: str, w: _Wanted) -> bool:
        """True when ``relpath`` respells this file's earlier path (``people/sam`` renamed to
        ``people/Sam``): the earlier one is stale, so its name can be used again."""
        old = self.by_key.get(key)
        return (
            old is not None
            and old.relpath != relpath
            and old.relpath not in self.claimed
            and (old.media_id, old.role) == (w.media.id, w.role)
            and _disk_state(self.project_dir, old) != "edited"  # an edited file stays put
        )


@dataclass
class _Writer:
    conn: sqlite3.Connection
    project_dir: Path
    hardlinks: bool
    occurrences: dict[int, list[_Occurrence]]
    written: int = 0
    linked: int = 0
    link_fallbacks: int = 0
    bytes_written: int = 0
    problems: list[ExportProblem] = field(default_factory=list)
    _links_work: bool = True
    _pending: list[_Record] = field(default_factory=list)

    def extract(self, zips: ZipCache, w: _Wanted) -> None:
        occurrences = self.occurrences[w.media.id]
        found = None
        for occ in occurrences:
            found = zips.member(occ.source_id, occ.member_index)
            if found is not None:
                break
        if found is None:
            names = sorted({zips.missing.get(o.source_id, o.label) for o in occurrences})
            self._problem(
                w,
                f"its zip ({', '.join(names)}) is missing or has changed since it was indexed; "
                "run `tripsort ingest` with the zip's current location",
            )
            return
        source, member = found
        target = _abs(self.project_dir, w.relpath)
        if self._occupied(target, w):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}{TMP_SUFFIX}")
        digest = hashlib.sha256()
        try:
            with tmp.open("wb") as out:
                for chunk in source.iter_chunks(member):
                    digest.update(chunk)
                    out.write(chunk)
        except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
            tmp.unlink(missing_ok=True)
            self._problem(w, f"could not be read from {source.path.name} ({exc})")
            return
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if digest.hexdigest() != w.media.sha256:  # pragma: no cover - needs a forged zip
            tmp.unlink(missing_ok=True)
            self._problem(w, f"its bytes in {source.path.name} no longer match the index")
            return
        _set_mtime(tmp, w.media.taken_at)
        tmp.replace(target)
        self._done(_record_for(self.project_dir, w, "extract"))

    def link(self, w: _Wanted) -> None:
        assert w.link_to is not None
        source = _abs(self.project_dir, w.link_to)
        if not source.is_file():
            self._problem(w, f"{w.link_to} is missing, so it cannot be linked")
            return
        target = _abs(self.project_dir, w.relpath)
        if self._occupied(target, w):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.hardlinks and self._links_work:
            try:
                target.hardlink_to(source)
            except FileExistsError:  # pragma: no cover - checked just above
                raise
            except OSError:  # exFAT, FAT32, and some network drives: copy from now on
                self._links_work = False
            else:
                self.linked += 1
                self._done(_record_for(self.project_dir, w, "hardlink"), count_bytes=False)
                return
        if self.hardlinks:  # wanted a hardlink but the drive refused
            self.link_fallbacks += 1
        tmp = target.with_name(f".{target.name}{TMP_SUFFIX}")
        shutil.copyfile(source, tmp)
        _set_mtime(tmp, w.media.taken_at)
        tmp.replace(target)
        self._done(_record_for(self.project_dir, w, "copy"))

    def _occupied(self, target: Path, w: _Wanted) -> bool:
        """A file appeared where the plan expected none (or an old spelling of the path could
        not be removed): leave it alone rather than overwrite it."""
        if not os.path.lexists(target):
            return False
        self._problem(w, f"{w.relpath} already exists and was left as it is")
        return True

    def _done(self, record: _Record, *, count_bytes: bool = True) -> None:
        self.written += 1
        if count_bytes:
            self.bytes_written += record.bytes
        self._pending.append(record)
        if len(self._pending) >= _COMMIT_EVERY:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            with transaction(self.conn):
                for record in self._pending:
                    _save_record(self.conn, record)
            self._pending.clear()

    def _problem(self, w: _Wanted, message: str) -> None:
        first = self.occurrences[w.media.id][0]
        self.problems.append(
            ExportProblem(w.media.sha256, f"{first.label}: {first.member_path}", message)
        )


def _abs(project_dir: Path, relpath: str) -> Path:
    return project_dir.joinpath(*relpath.split("/"))


def _disk_state(project_dir: Path, record: _Record) -> str:
    try:
        st = _abs(project_dir, record.relpath).stat()
    except FileNotFoundError:
        return "write"
    return (
        "unchanged" if (st.st_size, st.st_mtime_ns) == (record.bytes, record.mtime_ns) else "edited"
    )


def _same_content(path: Path, media: _Media) -> bool:
    if not path.is_file() or path.stat().st_size != media.bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest() == media.sha256


def _record_for(project_dir: Path, w: _Wanted, method: str) -> _Record:
    st = _abs(project_dir, w.relpath).stat()
    return _Record(
        relpath=w.relpath,
        fold_key=fold_key(w.relpath),
        media_id=w.media.id,
        role=w.role,
        method=method,
        bytes=st.st_size,
        mtime_ns=st.st_mtime_ns,
    )


def _save_record(conn: sqlite3.Connection, record: _Record) -> None:
    conn.execute(
        "INSERT INTO exported_files (relpath, fold_key, media_id, role, method, bytes, mtime_ns) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (relpath) DO UPDATE SET "
        "media_id = excluded.media_id, role = excluded.role, method = excluded.method, "
        "bytes = excluded.bytes, mtime_ns = excluded.mtime_ns",
        (
            record.relpath,
            record.fold_key,
            record.media_id,
            record.role,
            record.method,
            record.bytes,
            record.mtime_ns,
        ),
    )


def _remove_temp_files(project_dir: Path) -> None:
    """Delete temporary files left by an interrupted export."""
    for folder in _MANAGED_DIRS:
        root = project_dir / folder
        if root.is_dir():
            for tmp in root.rglob(f"*{TMP_SUFFIX}"):
                tmp.unlink(missing_ok=True)


def _remove_stale(
    conn: sqlite3.Connection, project_dir: Path, stale: list[_Record]
) -> tuple[int, int]:
    """Remove files an earlier export wrote that are no longer wanted.

    Returns (removed, left alone). Files changed since they were written are left alone and
    forgotten, so they are never touched again.
    """
    removed = left_alone = 0
    folders: set[Path] = set()
    with transaction(conn):
        for record in stale:
            path = _abs(project_dir, record.relpath)
            try:
                st = path.stat()
            except FileNotFoundError:
                conn.execute("DELETE FROM exported_files WHERE relpath = ?", (record.relpath,))
                continue
            if (st.st_size, st.st_mtime_ns) != (record.bytes, record.mtime_ns):
                left_alone += 1
            else:
                try:
                    path.unlink()
                except OSError:  # e.g. open in a viewer on Windows; try again next time
                    left_alone += 1
                    continue
                removed += 1
                folders.add(path.parent)
            conn.execute("DELETE FROM exported_files WHERE relpath = ?", (record.relpath,))
    for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        _prune_empty(folder, project_dir)
    return removed, left_alone


def _prune_empty(folder: Path, project_dir: Path) -> None:
    """Remove empty review folders, up to (and including) duplicates/ or unreadable/."""
    keep = {project_dir, project_dir / ALL_DIR}
    while folder not in keep and project_dir in folder.parents:
        try:
            folder.rmdir()
        except OSError:
            return
        folder = folder.parent


def _set_mtime(path: Path, taken_at: str | None) -> None:
    """Give the file its photo's date, read as wall-clock time on this machine."""
    if not taken_at:
        return
    try:
        stamp = int(datetime.fromisoformat(taken_at).timestamp() * 1_000_000_000)
        os.utime(path, ns=(stamp, stamp))
    except (ValueError, OverflowError, OSError):
        pass


def _load_media(conn: sqlite3.Connection) -> dict[int, _Media]:
    return {
        int(row[0]): _Media(int(row[0]), str(row[1]), int(row[2]), str(row[3]), row[4])
        for row in conn.execute("SELECT id, sha256, bytes, status, taken_at FROM media")
    }


def _load_occurrences(conn: sqlite3.Connection) -> dict[int, list[_Occurrence]]:
    found: defaultdict[int, list[_Occurrence]] = defaultdict(list)
    for media_id, source_id, member_index, member_path, label in conn.execute(
        "SELECT o.media_id, o.source_id, o.member_index, o.member_path, s.label "
        "FROM occurrences o JOIN sources s ON s.id = o.source_id "
        "ORDER BY s.label, o.member_path, o.member_index"
    ):
        found[int(media_id)].append(
            _Occurrence(int(source_id), int(member_index), str(member_path), str(label))
        )
    return dict(found)


def _load_copy_groups(conn: sqlite3.Connection) -> dict[int, list[int]]:
    """keeper id -> copy ids, copies in date order."""
    keepers: dict[int, int] = {}
    copies: defaultdict[int, list[int]] = defaultdict(list)
    for group_id, media_id, is_keeper in conn.execute(
        "SELECT gm.group_id, gm.media_id, gm.is_keeper FROM group_members gm "
        "JOIN media_groups g ON g.id = gm.group_id JOIN media m ON m.id = gm.media_id "
        "WHERE g.kind = 'copy' ORDER BY COALESCE(m.taken_at, ''), m.sha256"
    ):
        if is_keeper:
            keepers[int(group_id)] = int(media_id)
        else:
            copies[int(group_id)].append(int(media_id))
    return {keepers[g]: members for g, members in copies.items() if g in keepers}


def _load_records(conn: sqlite3.Connection) -> dict[str, _Record]:
    return {
        str(row[0]): _Record(
            str(row[0]),
            str(row[1]),
            int(row[2]),
            str(row[3]),
            str(row[4]),
            int(row[5]),
            int(row[6]),
        )
        for row in conn.execute(
            "SELECT relpath, fold_key, media_id, role, method, bytes, mtime_ns FROM exported_files"
        )
    }


def _load_people_folders(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """media id -> the people/ folders it belongs in (anyone whose face it shows now)."""
    found: defaultdict[int, set[str]] = defaultdict(set)
    for media_id, number, name in conn.execute(
        "SELECT DISTINCT f.media_id, p.number, p.name FROM faces f "
        "JOIN people p ON p.id = f.person_id WHERE f.match IS NOT NULL"
    ):
        found[int(media_id)].add(person_folder(display_name(int(number), name)))
    return {media_id: sorted(folders) for media_id, folders in found.items()}


def _load_shot_folders(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """media id -> its shots/ folders: solo or group, plus selfie."""
    found: dict[int, list[str]] = {}
    for media_id, shot, is_selfie in conn.execute(
        "SELECT id, shot, is_selfie FROM media WHERE shot IN ('solo', 'group')"
    ):
        found[int(media_id)] = [str(shot), "selfie"] if is_selfie else [str(shot)]
    return found
