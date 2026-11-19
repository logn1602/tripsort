"""Dedupe metrics.

Every item (a zip member) has a true origin (the photo it came from) and a predicted
cluster (the keeper it was grouped under, or itself when it is kept in all/).

The headline numbers measure what a user sees in all/:

* precision: of the files left out of all/, the share whose photo is still in all/
  (a miss here means a distinct photo vanished from all/)
* recall: of the copies, the share left out of all/ (a miss leaves a duplicate in all/)

Pairwise grouping numbers are diagnostics: they also count which keeper a copy was filed
under, which does not change what ends up in all/.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    member: str
    origin: str
    role: str  # original, copy, negative
    variant: str
    cluster: str  # predicted: id of the keeper it was grouped under (itself if alone)
    distance: int | None = None  # pHash distance to its origin's original, for copies


def kept(item: Item) -> bool:
    return item.cluster == item.member


def source_photo_merges(items: list[Item]) -> int:
    """Source photos (role "original") filed under a different source photo.

    With real photos as sources this is ambiguous: two of them may really be copies of each
    other (a WhatsApp copy in one folder of a photo in another), which only a person can tell.
    """
    by_cluster: defaultdict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.role == "original":
            by_cluster[item.cluster].add(item.origin)
    return sum(len(origins) - 1 for origins in by_cluster.values())


def merge_source_photos(items: list[Item]) -> list[Item]:
    """The items with origins joined wherever source photos share a cluster: the optimistic
    reading, in which every such merge was a real copy."""
    parent: dict[str, str] = {}

    def find(origin: str) -> str:
        while parent.get(origin, origin) != origin:
            origin = parent[origin]
        return origin

    first: dict[str, str] = {}
    for item in items:
        if item.role == "original":
            if item.cluster in first:
                parent[find(item.origin)] = find(first[item.cluster])
            else:
                first[item.cluster] = item.origin
    return [Item(i.member, find(i.origin), i.role, i.variant, i.cluster, i.distance) for i in items]


def outcome(items: list[Item]) -> dict[str, float | int]:
    """Precision and recall of what is left out of all/ (see the module docstring)."""
    kept_origins = {item.origin for item in items if kept(item)}
    left_out = [item for item in items if not kept(item)]
    wrongly = [item for item in left_out if item.origin not in kept_origins]
    copies = [item for item in items if item.role == "copy"]
    caught = [item for item in copies if not kept(item) and item.origin in kept_origins]
    return {
        "precision": round(1 - len(wrongly) / len(left_out), 4) if left_out else 1.0,
        "recall": round(len(caught) / len(copies), 4) if copies else 1.0,
        "left_out": len(left_out),
        "wrongly_left_out": len(wrongly),
        "copies": len(copies),
        "copies_left_in_all": sum(1 for item in copies if kept(item)),
    }


def wrongly_left_out_by_family(items: list[Item]) -> dict[str, float]:
    """For originals and each hard-negative family: the share whose photo vanished from all/."""
    kept_origins = {item.origin for item in items if kept(item)}
    lost: defaultdict[str, list[bool]] = defaultdict(list)
    for item in items:
        if item.role in {"original", "negative"}:
            lost[item.variant].append(item.origin not in kept_origins)
    return {family: round(sum(hits) / len(hits), 4) for family, hits in sorted(lost.items())}


def copies_left_in_all_by_variant(items: list[Item]) -> dict[str, float]:
    """For each copy type: the share still in all/ (missed copies)."""
    missed: defaultdict[str, list[bool]] = defaultdict(list)
    for item in items:
        if item.role == "copy":
            missed[item.variant].append(kept(item))
    return {variant: round(sum(hits) / len(hits), 4) for variant, hits in sorted(missed.items())}


def _pairs(counts: Iterable[int]) -> int:
    return sum(n * (n - 1) // 2 for n in counts)


def pairwise(items: list[Item]) -> dict[str, float]:
    together = _pairs(Counter(item.cluster for item in items).values())
    same_origin = _pairs(Counter(item.origin for item in items).values())
    correct = _pairs(Counter((item.cluster, item.origin) for item in items).values())
    precision = correct / together if together else 1.0
    recall = correct / same_origin if same_origin else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def recall_by_variant(items: list[Item]) -> dict[str, float]:
    """For each copy type: the share of copies grouped with their own original."""
    original_cluster = {i.origin: i.cluster for i in items if i.role == "original"}
    found: defaultdict[str, list[bool]] = defaultdict(list)
    for item in items:
        if item.role == "copy":
            found[item.variant].append(item.cluster == original_cluster.get(item.origin))
    return {variant: round(sum(hits) / len(hits), 4) for variant, hits in sorted(found.items())}


def false_merges_by_family(items: list[Item]) -> dict[str, float]:
    """For each hard-negative family: the share of its items grouped with anything else."""
    origins_in: defaultdict[str, set[str]] = defaultdict(set)
    for item in items:
        origins_in[item.cluster].add(item.origin)
    merged: defaultdict[str, list[bool]] = defaultdict(list)
    for item in items:
        if item.role == "negative":
            merged[item.variant].append(len(origins_in[item.cluster]) > 1)
    return {family: round(sum(hits) / len(hits), 4) for family, hits in sorted(merged.items())}


def keeper_accuracy(items: list[Item]) -> float:
    """Among originals grouped with at least one copy, the share that were chosen as keeper."""
    by_cluster: defaultdict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_cluster[item.cluster].append(item)
    hits = total = 0
    for item in items:
        if item.role != "original":
            continue
        group = by_cluster[item.cluster]
        if any(other.role == "copy" and other.origin == item.origin for other in group):
            total += 1
            hits += item.cluster == item.member
    return round(hits / total, 4) if total else 1.0


def distance_histogram(distances: Iterable[int | None], cap: int = 12) -> dict[str, int]:
    """Counts per pHash distance, with everything above ``cap`` in one bucket."""
    counts: Counter[str] = Counter()
    for d in distances:
        if d is None:
            counts["none"] += 1
        else:
            counts[str(d) if d <= cap else f">{cap}"] += 1
    order = [str(d) for d in range(cap + 1)] + [f">{cap}", "none"]
    return {key: counts[key] for key in order if counts[key]}
