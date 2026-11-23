"""The people API: list people, rename and merge them, name them from refs/, review them,
and see or dismiss merge suggestions.

Renames, merges, and dismissed suggestions are stored in the index and pin the people
involved: the next analysis keeps them, so an export after a rename only moves links and
reads no pixels.
"""

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tripsort.commands import command
from tripsort.errors import PeopleError
from tripsort.faces.names import check_name, display_name, name_key
from tripsort.faces.suggest import Suggestion, suggest_merges
from tripsort.index.db import transaction
from tripsort.progress import Progress

if TYPE_CHECKING:
    from tripsort.api import AnalyzeResult, Project

_NUMBER = re.compile(r"^person_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class PersonInfo:
    name: str  # 'alex', or 'person_07' while unnamed
    number: int  # person_07 keeps 7 even after a rename
    named_by: str | None  # 'refs', 'user', or None
    photos: int
    solo: int  # photos where they are the only foreground face
    faces: int
    videos: int = 0  # videos they are in (matched in enough sampled frames)


@dataclass
class RefsResult:
    # ref name -> 'named' (a person has it), 'no match' (nobody similar enough),
    # 'no usable photo', or 'named by you' (you gave the name, or its best match, a name)
    names: dict[str, str]
    problems: list[str]
    analysis: "AnalyzeResult"

    def summary(self) -> str:
        named = sorted(name for name, state in self.names.items() if state == "named")
        lines = [f"refs/ named {len(named)} of {len(self.names)} people."]
        for state in ("no match", "no usable photo", "named by you"):
            names = sorted(name for name, s in self.names.items() if s == state)
            if names:
                lines.append(f"  {state}: {', '.join(names)}")
        return "\n".join(lines)


class People:
    """``Project.people``: the people found in a project."""

    def __init__(self, project: "Project") -> None:
        self._project = project

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._project.connection

    def all(self) -> list[PersonInfo]:
        """Everyone in at least one photo, plus named people, most photos first."""
        self._fresh()
        people = [p for p in self._infos() if p.photos or p.named_by]
        return sorted(people, key=lambda p: (-p.photos, p.number))

    def get(self, person: str) -> PersonInfo:
        """A person by name or by number ('person_07'); ``PeopleError`` if there is none."""
        self._fresh()
        row = self._find(person)
        return self._infos(int(row["id"]))[0]

    def rename(self, person: str, new_name: str) -> PersonInfo:
        """Give ``person`` a name. A name refs/ gave someone else moves to this person; a name
        you gave someone else is refused (merge them instead)."""
        new = check_name(new_name)
        key = name_key(new)
        self._fresh()
        with transaction(self._conn):
            target = self._find(person)
            holder = self._conn.execute(
                "SELECT id, name, named_by FROM people WHERE name_key = ?", (key,)
            ).fetchone()
            if holder is not None and int(holder[0]) != int(target["id"]):
                if holder[2] == "user":
                    raise PeopleError(
                        f"{holder[1]!r} is the name of someone else; if they are the same "
                        f"person, run `{command('people merge', 'PROJECT', person, holder[1])}`"
                    )
                self._conn.execute(
                    "UPDATE people SET name = NULL, name_key = NULL, named_by = NULL WHERE id = ?",
                    (holder[0],),
                )
            self._conn.execute(
                "UPDATE people SET name = ?, name_key = ?, named_by = 'user', pinned = 1 "
                "WHERE id = ?",
                (new, key, target["id"]),
            )
            self._project._people_edited()
        return self._infos(int(target["id"]))[0]

    def merge(self, person: str, into: str) -> PersonInfo:
        """Say that ``person`` and ``into`` are one person: every face of ``person`` moves to
        ``into``, which keeps its name (or takes ``person``'s, if it had none)."""
        self._fresh()
        with transaction(self._conn):
            source = self._find(person)
            target = self._find(into)
            if source["id"] == target["id"]:
                raise PeopleError("a person cannot be merged into themselves")
            self._conn.execute(
                "UPDATE faces SET person_id = ? WHERE person_id = ?", (target["id"], source["id"])
            )
            self._conn.execute("DELETE FROM people WHERE id = ?", (source["id"],))
            if target["name"] is None and source["name"] is not None:
                self._conn.execute(
                    "UPDATE people SET name = ?, name_key = ?, named_by = ? WHERE id = ?",
                    (source["name"], source["name_key"], source["named_by"], target["id"]),
                )
            self._conn.execute("UPDATE people SET pinned = 1 WHERE id = ?", (target["id"],))
            self._project._people_edited()
        return self._infos(int(target["id"]))[0]

    def suggestions(self) -> list[Suggestion]:
        """Unnamed people who may be a named person split in two, most similar first. Nothing
        is merged: ``merge`` the ones that are, and ``dismiss`` the others."""
        self._fresh()
        return suggest_merges(self._conn, self._project.config.people.suggest_min_similarity)

    def dismiss(self, person: str, other: str) -> tuple[PersonInfo, PersonInfo]:
        """Say ``person`` and ``other`` are two people: merging them is never suggested again.
        Both are pinned, as a merge pins, so later clusterings keep them apart."""
        self._fresh()
        with transaction(self._conn):
            first = self._find(person)
            second = self._find(other)
            if first["id"] == second["id"]:
                raise PeopleError("a person cannot be kept apart from themselves")
            low, high = sorted((int(first["id"]), int(second["id"])))
            self._conn.execute(
                "INSERT OR IGNORE INTO dismissed_suggestions (person_id, other_id) VALUES (?, ?)",
                (low, high),
            )
            self._conn.execute("UPDATE people SET pinned = 1 WHERE id IN (?, ?)", (low, high))
            self._project._people_edited()
        return self._infos(int(first["id"]))[0], self._infos(int(second["id"]))[0]

    def name_from_refs(self, folder: str | os.PathLike[str]) -> RefsResult:
        """Name people from ``folder`` (``<name>.jpg`` or ``<name>/*.jpg``), then analyze.

        The photos replace any earlier refs, and later analyses keep using them. Needs the
        face models; raises ``ModelError`` when they cannot be loaded.
        """
        from tripsort.faces.naming import store_refs

        report = store_refs(self._conn, Path(folder), self._project.face_analyzer())
        analysis = self._project.analyze()
        named = {
            str(name)
            for (name,) in self._conn.execute(
                "SELECT name FROM people WHERE named_by = 'refs' AND name IS NOT NULL"
            )
        }
        unmatched = set(analysis.people.refs_unmatched)
        names: dict[str, str] = {}
        for name, usable in report.names.items():
            if not usable:
                names[name] = "no usable photo"
            elif name in named:
                names[name] = "named"
            elif name in unmatched:
                names[name] = "no match"
            else:
                names[name] = "named by you"
        return RefsResult(names, report.problems, analysis)

    def review(
        self,
        *,
        per_person: int = 24,
        unassigned: int = 120,
        progress: Progress | None = None,
        project: str | None = None,
    ) -> Path:
        """Write ``review/people.html``: the merge suggestions with every face of the person
        suggested, sample face crops per person, read back from the zips, and faces that
        belong to nobody. ``project`` is how the page's commands name the project folder
        (default: its full path). Returns the page's path."""
        from tripsort.faces.review import write_review

        suggestions = self.suggestions()
        return write_review(
            self._conn,
            self._project.folder,
            per_person=per_person,
            unassigned=unassigned,
            progress=progress,
            suggestions=suggestions,
            project=project if project is not None else str(self._project.folder),
        )

    def _fresh(self) -> None:
        """People come from the last analysis: analyze first when the index changed since."""
        if self._project.analysis_is_stale():
            self._project.analyze()

    def _find(self, person: str) -> dict[str, Any]:
        text = " ".join(person.split())
        number = _NUMBER.match(text)
        if number:
            row = self._conn.execute(
                "SELECT id, number, name, name_key, named_by FROM people WHERE number = ?",
                (int(number.group(1)),),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id, number, name, name_key, named_by FROM people WHERE name_key = ?",
                (name_key(text),),
            ).fetchone()
        if row is None:
            raise PeopleError(
                f"nobody is called {text!r}; `{command('people list', 'PROJECT')}` shows everyone"
            )
        return dict(zip(("id", "number", "name", "name_key", "named_by"), row, strict=True))

    def _infos(self, person_id: int | None = None) -> list[PersonInfo]:
        return person_infos(self._conn, person_id)


def person_infos(conn: sqlite3.Connection, person_id: int | None = None) -> list[PersonInfo]:
    """Every person (or one), with their photo and face counts as of the last analysis."""
    where = "WHERE p.id = ?" if person_id is not None else ""
    rows = conn.execute(
        "SELECT p.number, p.name, p.named_by, "
        "COUNT(DISTINCT CASE WHEN f.frame_ts = -1 THEN f.media_id END), "
        "COUNT(DISTINCT CASE WHEN m.shot = 'solo' AND f.match != 'background' "
        "                    AND f.frame_ts = -1 THEN f.media_id END), "
        "COUNT(CASE WHEN f.frame_ts = -1 THEN f.id END), "
        "COUNT(DISTINCT CASE WHEN f.frame_ts >= 0 THEN f.media_id END) FROM people p "
        "LEFT JOIN faces f ON f.person_id = p.id AND f.match IS NOT NULL "
        f"LEFT JOIN media m ON m.id = f.media_id {where} "
        "GROUP BY p.id, p.number, p.name, p.named_by",
        () if person_id is None else (person_id,),
    ).fetchall()
    return [
        PersonInfo(
            name=display_name(int(number), name),
            number=int(number),
            named_by=named_by,
            photos=int(photos),
            solo=int(solo),
            faces=int(faces),
            videos=int(videos),
        )
        for number, name, named_by, photos, solo, faces, videos in rows
    ]
