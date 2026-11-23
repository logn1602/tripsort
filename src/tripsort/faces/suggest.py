"""Merge suggestions: unnamed people who may be a named person, split in two by the clustering.

A friend can come out of the clustering as two people, for example with and without
sunglasses, and a ref names only one of them. tripsort suggests the merge and never makes
it: on the labeled trip, strangers who looked like a friend were as close to the friend's
average face as the friend's own second cluster (0.68 against 0.70), so only the user can
tell them apart, with ``tripsort people review``. A merge, or a dismissed suggestion, is
then the user's own decision and pins the people involved.

An unnamed person is suggested for a named one when

* their average faces are at least ``suggest_min_similarity`` similar;
* they share no photo, since two faces in one photo are two people (the named person's
  background matches count);
* and the user has not dismissed the pair.

The most similar named person is suggested, and the next one as well when it is within
``SECOND_WITHIN`` of the first: two friends who look alike.
"""

import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Set
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tripsort.commands import command
from tripsort.faces.names import display_name
from tripsort.faces.types import EMBEDDING_SIZE
from tripsort.stats import count

Vector = npt.NDArray[np.float32]
SECOND_WITHIN = 0.05  # a second candidate this close to the first is shown too


@dataclass(frozen=True)
class Candidate:
    name: str
    number: int
    similarity: float  # of the two people's average faces


@dataclass(frozen=True)
class Suggestion:
    person: str  # 'person_07': suggestions are only made for unnamed people
    number: int
    photos: int
    candidates: tuple[Candidate, ...]  # most similar first; two when they are close


def pick_candidates(
    unnamed: Mapping[int, Vector],
    named: Mapping[int, Vector],
    photos: Mapping[int, Set[int]],
    dismissed: Set[frozenset[int]],
    min_similarity: float,
    second_within: float = SECOND_WITHIN,
) -> dict[int, list[tuple[int, float]]]:
    """For each unnamed person (id -> average face), the named people they may be, as
    ``(person id, similarity)``, most similar first. ``photos`` holds each person's photos,
    and ``dismissed`` the pairs the user said are two people."""
    if not unnamed or not named:
        return {}
    named_ids = sorted(named)
    matrix = np.stack([named[p] for p in named_ids])
    picked: dict[int, list[tuple[int, float]]] = {}
    for person in sorted(unnamed):
        own = photos.get(person, frozenset())
        similarities = (matrix @ unnamed[person]).tolist()
        eligible = sorted(
            (
                (float(similarity), other)
                for other, similarity in zip(named_ids, similarities, strict=True)
                if similarity >= min_similarity
                and not own & photos.get(other, frozenset())
                and frozenset((person, other)) not in dismissed
            ),
            key=lambda c: (-c[0], c[1]),
        )
        if not eligible:
            continue
        chosen = eligible[:1]
        if len(eligible) > 1 and eligible[0][0] - eligible[1][0] <= second_within:
            chosen.append(eligible[1])
        picked[person] = [(other, similarity) for similarity, other in chosen]
    return picked


def suggest_merges(
    conn: sqlite3.Connection, min_similarity: float, second_within: float = SECOND_WITHIN
) -> list[Suggestion]:
    """The suggestions for the people of the last analysis, most similar first."""
    people = {
        int(pid): (int(number), name, centroid)
        for pid, number, name, centroid in conn.execute(
            "SELECT id, number, name, centroid FROM people"
        )
    }
    photos: defaultdict[int, set[int]] = defaultdict(set)
    for pid, media_id in conn.execute(
        "SELECT person_id, media_id FROM faces WHERE person_id IS NOT NULL AND match IS NOT NULL"
    ):
        photos[int(pid)].add(int(media_id))
    vectors = {
        pid: np.frombuffer(blob, dtype=np.float32, count=EMBEDDING_SIZE)
        for pid, (_, _, blob) in people.items()
        if blob is not None and photos.get(pid)
    }
    unnamed = {pid: v for pid, v in vectors.items() if not people[pid][1]}
    named = {pid: v for pid, v in vectors.items() if people[pid][1]}
    dismissed = {
        frozenset((int(a), int(b)))
        for a, b in conn.execute("SELECT person_id, other_id FROM dismissed_suggestions")
    }
    picked = pick_candidates(unnamed, named, photos, dismissed, min_similarity, second_within)
    suggestions = [
        Suggestion(
            person=display_name(people[pid][0], None),
            number=people[pid][0],
            photos=len(photos[pid]),
            candidates=tuple(
                Candidate(str(people[other][1]), people[other][0], similarity)
                for other, similarity in candidates
            ),
        )
        for pid, candidates in picked.items()
    ]
    suggestions.sort(key=lambda s: (-s.candidates[0].similarity, -s.photos, s.number))
    return suggestions


def people_command(action: str, project: str, *people: str) -> str:
    """``tripsort people ACTION PROJECT PEOPLE...``, ready to paste (see ``command``)."""
    return command(f"people {action}", project, *people)


def describe(suggestion: Suggestion) -> str:
    """'person_07 (45 photos) may be Sam (similarity 0.70)'."""
    names = " or ".join(f"{c.name} (similarity {c.similarity:.2f})" for c in suggestion.candidates)
    return f"{suggestion.person} ({count(suggestion.photos, 'photo')}) may be {names}"


def commands(suggestion: Suggestion, project: str) -> list[tuple[str, str]]:
    """``('merge', command)`` for each candidate, then ``('dismiss', command)`` for each."""
    return [
        (action, people_command(action, project, suggestion.person, c.name))
        for action in ("merge", "dismiss")
        for c in suggestion.candidates
    ]


def suggestion_lines(
    suggestions: list[Suggestion], project: str, limit: int | None = None
) -> list[str]:
    """The suggestions as summary lines, each with the exact commands to merge or dismiss."""
    if not suggestions:
        return []
    review = people_command("review", project)
    lines = [
        f"Maybe the same person ({count(len(suggestions), 'suggestion')}); "
        f"check `{review}` before merging:"
    ]
    shown = suggestions if limit is None else suggestions[:limit]
    for suggestion in shown:
        lines.append(f"  {describe(suggestion)}")
        lines.extend(
            f"    {action + ':':<9}{command}" for action, command in commands(suggestion, project)
        )
    if len(shown) < len(suggestions):
        lines.append(f"  {len(suggestions) - len(shown)} more: {people_command('list', project)}")
    return lines
