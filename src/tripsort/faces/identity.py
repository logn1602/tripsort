"""Keep people's identities when faces are clustered again.

Clusters are rebuilt from scratch whenever faces or settings change, so a cluster label means
nothing from one analysis to the next. People persist instead: every face remembers the last
person it belonged to, and each new cluster becomes the person most of its faces belonged to.

* A person the user confirmed (renamed, or merged others into) is *pinned*: every cluster
  that is mostly theirs stays theirs, so a merge survives the next clustering.
* Any other person goes to the one cluster holding most of their faces, and further clusters
  that were mostly theirs become new people. That way a stricter threshold can still split
  a person who was really two.
* When one cluster holds at least ``min_faces`` faces of each of two or more pinned people,
  it is split between them (each face by its last person, the others by the closest of
  their average faces): clustering never undoes what the user confirmed.
* Clusters without history become new people.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from tripsort.faces.cluster import NOISE, Embeddings, Labels
from tripsort.faces.types import normalized


@dataclass
class CarryOver:
    # cluster label -> the existing person it continues, or None for a new person
    cluster_person: dict[int, int | None] = field(default_factory=dict)
    # face position -> person, for the faces of clusters split between pinned people
    face_person: dict[int, int] = field(default_factory=dict)
    # pinned people found together in one cluster, which was split between them
    joined: list[tuple[int, ...]] = field(default_factory=list)

    def person_of(self, position: int, label: int) -> int | None:
        if position in self.face_person:
            return self.face_person[position]
        return self.cluster_person.get(label)


def carry_over(
    labels: Labels,
    previous: Sequence[int | None],
    pinned: set[int],
    embeddings: Embeddings,
    *,
    min_faces: int,
) -> CarryOver:
    """Map new clusters to existing people. ``previous`` is each face's last person."""
    members: defaultdict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(labels):
        if label != NOISE:
            members[int(label)].append(position)
    result = CarryOver()
    claims: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for label in sorted(members):
        faces = members[label]
        history = Counter(p for p in (previous[i] for i in faces) if p is not None)
        if not history:
            result.cluster_person[label] = None
            continue
        pinned_here = sorted(p for p, n in history.items() if p in pinned and n >= min_faces)
        if len(pinned_here) >= 2:
            _split(faces, previous, pinned_here, embeddings, result)
            result.joined.append(tuple(pinned_here))
            continue
        # most faces, then a pinned person, then the oldest person
        best = max(history, key=lambda p: (history[p], p in pinned, -p))
        claims[best].append((history[best], len(faces), label))
    for person, claimed in claims.items():
        if person in pinned:
            for _, _, label in claimed:
                result.cluster_person[label] = person
            continue
        claimed.sort(key=lambda c: (-c[0], -c[1], c[2]))
        result.cluster_person[claimed[0][2]] = person
        for _, _, label in claimed[1:]:
            result.cluster_person[label] = None
    return result


def _split(
    faces: list[int],
    previous: Sequence[int | None],
    people: list[int],
    embeddings: Embeddings,
    result: CarryOver,
) -> None:
    matrix = np.stack(
        [
            normalized(embeddings[[i for i in faces if previous[i] == p]].mean(axis=0))
            for p in people
        ]
    )
    for i in faces:
        last = previous[i]
        if last is not None and last in people:
            result.face_person[i] = last
        else:
            result.face_person[i] = people[int(np.argmax(matrix @ embeddings[i]))]
