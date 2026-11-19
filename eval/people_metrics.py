"""People metrics on photo-level labels, plus face-level ones taken free from solo photos.

Predicted people are matched to labeled people by name first (refs/ and renames give real
names), then one to one by the most shared photos. A person split in two keeps only their
bigger part, so the other part counts as missed photos: that is what a user sees, one
folder per person.

* precision per person: of the photos in their folder, the share they are really in
  (protect it first: a wrong photo in a friend's folder is the worst mistake);
* recall per person: of the photos they are in, the share that made it into their folder;
* homogeneity and completeness on faces: from solo photos (one labeled person, one
  foreground face), where the face's true person is known without labeling faces.

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


def match_people(photos: list[Photo]) -> Matching:
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


def people_report(photos: list[Photo]) -> dict[str, Any]:
    matching = match_people(photos)
    counts: Counter[str] = Counter(t for p in photos for t in p.labeled)
    alias = {
        key: f"p{i}"
        for i, (key, _) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), 1)
    }
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    unmatched_photos = 0
    for photo in photos:
        mapped = {matching.pairs[p] for p in photo.predicted if p in matching.pairs}
        if photo.predicted - set(matching.pairs):
            unmatched_photos += 1
        for t in photo.labeled & mapped:
            tp[t] += 1
        for t in mapped - photo.labeled:
            fp[t] += 1
        for t in photo.labeled - mapped:
            fn[t] += 1
    per_person = []
    for key in sorted(counts, key=lambda k: alias[k]):
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
        "labeled_people": len(counts),
        "predicted_people": len(predicted_people),
        "matched_by_name": matching.by_name,
        "matched_by_photos": len(matching.pairs) - matching.by_name,
        "micro": {
            "precision": _ratio(total_tp, total_tp + total_fp),
            "recall": _ratio(total_tp, total_tp + total_fn),
        },
        "macro": {
            "precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
            "recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        },
        "photos_with_unmatched_predicted_people": unmatched_photos,
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
