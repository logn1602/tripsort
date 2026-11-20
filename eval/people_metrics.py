"""People metrics on photo-level labels, plus face-level ones taken free from solo photos.

Only some people are labeled: the labeler and the friends they have refs for. They are
labeled in every checked row, so a checked row that leaves one of them out says they are
not in it. Every ref name is a labeled person, even one no checked row lists: tripsort
putting it on a checked photo is a wrong photo. Everyone else tripsort finds (strangers,
friends without a ref) is counted but never scored, since nobody labeled them.

Predicted people are matched to labeled people by name first (refs/ and renames give real
names), then one to one by the most shared photos. A person split in two keeps only their
bigger part, so the other part counts as missed photos: that is what a user sees, one
folder per person.

* precision per person: of the photos in their folder, the share they are really in
  (protect it first: a wrong photo in a friend's folder is the worst mistake);
* recall per person: of the photos they are in, the share that made it into their folder;
* homogeneity and completeness on faces: from solo photos of labeled people (labeled solo,
  one labeled person, one foreground face found), where the face's true person is known
  without labeling faces. Solo photos of anyone else are skipped. A solo photo of someone
  unlabeled with a labeled friend in the background would slip in; that should be rare.

Every name is replaced by p1, p2, ... (most labeled photos first) before results leave
this module, so they can be printed and saved.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import confusion_matrix, homogeneity_completeness_v_measure

SHOTS = ("solo", "group", "none")


@dataclass
class Photo:
    labeled: set[str]  # person keys (casefolded names)
    predicted: set[str]  # predicted people's keys
    shot: str  # labeled
    predicted_shot: str | None
    selfie: bool
    predicted_selfie: bool
    # for photos with one labeled person and one foreground face: the predicted person of
    # that face, or None when it is in nobody's folder
    solo_face: tuple[str, str | None] | None = None
    category: str = ""


@dataclass
class Matching:
    pairs: dict[str, str] = field(default_factory=dict)  # predicted key -> labeled key
    by_name: int = 0


def solo_face(
    labeled: set[str], shot: str, foreground: list[str | None]
) -> tuple[str, str | None] | None:
    """(true person, predicted person) of a solo photo's one face, when the true one is
    known: the labeler called it solo, named one person in it, and one foreground face was
    found. A group photo with one labeled person and one face found is skipped: the face
    may be someone unlabeled."""
    if shot != "solo" or len(labeled) != 1 or len(foreground) != 1:
        return None
    return next(iter(labeled)), foreground[0]


def match_people(photos: list[Photo], labeled: set[str] | None = None) -> Matching:
    """``labeled`` defaults to the people labeled in ``photos``."""
    if labeled is None:
        labeled = set().union(*(p.labeled for p in photos)) if photos else set()
    predicted = set().union(*(p.predicted for p in photos)) if photos else set()
    result = Matching({name: name for name in predicted & labeled})
    result.by_name = len(result.pairs)
    free_pred = predicted - set(result.pairs)
    free_true = labeled - set(result.pairs.values())
    shared: Counter[tuple[str, str]] = Counter()
    for photo in photos:
        for p in photo.predicted & free_pred:
            for t in photo.labeled & free_true:
                shared[(p, t)] += 1
    for (p, t), _ in sorted(shared.items(), key=lambda kv: (-kv[1], kv[0])):
        if p in free_pred and t in free_true:
            result.pairs[p] = t
            free_pred.discard(p)
            free_true.discard(t)
    return result


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def people_report(photos: list[Photo], everyone: Counter[str] | None = None) -> dict[str, Any]:
    """``everyone`` holds every labeled person with their photos over every checked row, both
    splits; a ref name no checked row lists is in it with 0. A labeled person is labeled in
    every checked row, so one predicted in a photo of this split that does not list them is
    a wrong photo, even with no labeled photo here."""
    counts: Counter[str] = Counter(t for p in photos for t in p.labeled)
    totals = everyone if everyone is not None else counts
    labeled = set(totals) | set(counts)
    matching = match_people(photos, labeled)
    order = sorted(labeled, key=lambda k: (-totals[k], k))
    alias = {key: f"p{i}" for i, key in enumerate(order, 1)}
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for photo in photos:
        mapped = {matching.pairs[p] for p in photo.predicted if p in matching.pairs}
        for t in photo.labeled & mapped:
            tp[t] += 1
        for t in mapped - photo.labeled:
            fp[t] += 1
        for t in photo.labeled - mapped:
            fn[t] += 1
    per_person = []
    for key in order:
        per_person.append(
            {
                "person": alias[key],
                "photos": counts[key],
                "matched": key in matching.pairs.values(),
                "precision": _ratio(tp[key], tp[key] + fp[key]),
                "recall": _ratio(tp[key], tp[key] + fn[key]),
            }
        )
    precisions = [p["precision"] for p in per_person if p["precision"] is not None]
    recalls = [p["recall"] for p in per_person if p["recall"] is not None]
    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    predicted_people = set().union(*(p.predicted for p in photos)) if photos else set()
    return {
        "labeled_people": len(labeled),
        "predicted_people": len(predicted_people),
        "matched_by_name": matching.by_name,
        "matched_by_photos": len(matching.pairs) - matching.by_name,
        # strangers and friends nobody labeled: counted, never scored
        "other_predicted_people": len(predicted_people - set(matching.pairs)),
        "micro": {
            "precision": _ratio(total_tp, total_tp + total_fp),
            "recall": _ratio(total_tp, total_tp + total_fn),
        },
        "macro": {
            "precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        },
        "per_person": per_person,
    }


def shot_report(photos: list[Photo]) -> dict[str, Any]:
    truth = [p.shot for p in photos]
    guess = [p.predicted_shot or "none" for p in photos]
    matrix = confusion_matrix(truth, guess, labels=list(SHOTS)).tolist() if photos else []
    correct = sum(t == g for t, g in zip(truth, guess, strict=True))
    selfie_tp = sum(p.selfie and p.predicted_selfie for p in photos)
    return {
        "accuracy": _ratio(correct, len(photos)),
        "confusion": {"labels": list(SHOTS), "rows_true_columns_predicted": matrix},
        "not_scanned": sum(p.predicted_shot is None for p in photos),
        "selfie": {
            "labeled": sum(p.selfie for p in photos),
            "precision": _ratio(selfie_tp, sum(p.predicted_selfie for p in photos)),
            "recall": _ratio(selfie_tp, sum(p.selfie for p in photos)),
        },
    }


def face_report(photos: list[Photo]) -> dict[str, Any]:
    faces = [p.solo_face for p in photos if p.solo_face is not None]
    assigned = [(t, g) for t, g in faces if g is not None]
    result: dict[str, Any] = {
        "solo_faces": len(faces),
        "unassigned": _ratio(len(faces) - len(assigned), len(faces)),
    }
    if assigned:
        h, c, v = homogeneity_completeness_v_measure(
            [t for t, _ in assigned], [g for _, g in assigned]
        )
        result.update(homogeneity=round(h, 4), completeness=round(c, 4), v_measure=round(v, 4))
    return result


def category_counts(photos: list[Photo]) -> dict[str, int]:
    return dict(sorted(Counter(p.category or "unlabeled" for p in photos).items()))
