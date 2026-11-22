"""Category metrics on photo-level labels: precision, recall, and F1 per category, the
confusion matrix on the primary category, and macro F1 over the labeled categories.

Category names are not private, so they appear as they are. A category nobody labeled a
photo with (no support) has no recall, but every photo tripsort puts in it is a wrong photo,
counted under ``wrong_photos``.
"""

from typing import Any

OTHER = "other"


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def category_report(pairs: list[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    """``pairs`` holds (labeled, predicted) for each scored photo; ``labels`` the categories
    in the order to report them (predicted labels not in it are added at the end)."""
    order = list(labels) + sorted({p for _, p in pairs} - set(labels))
    per: dict[str, dict[str, Any]] = {}
    for label in order:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None:
            f1 = round(2 * precision * recall / (precision + recall), 4) if tp else 0.0
        per[label] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "right": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    labeled = [label for label in order if label != OTHER and per[label]["support"]]
    f1s = [per[label]["f1"] or 0.0 for label in labeled]
    return {
        "photos": len(pairs),
        "accuracy": _ratio(sum(1 for t, p in pairs if t == p), len(pairs)),
        "macro_f1_labeled": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "labeled_categories": labeled,
        # photos put in a category nobody labeled any photo with: each one is a wrong photo
        "wrong_photos": {
            label: per[label]["predicted"]
            for label in order
            if label != OTHER and not per[label]["support"] and per[label]["predicted"]
        },
        "per_category": per,
        "confusion": {
            "labels": order,
            "rows_labeled_columns_predicted": [
                [sum(1 for t, p in pairs if t == a and p == b) for b in order] for a in order
            ],
        },
    }


def flips(before: list[int], after: list[int]) -> dict[str, int]:
    """How many rows went from right to wrong and from wrong to right, given the row numbers
    each run got right (the int8 keep rule allows at most one right-to-wrong)."""
    a, b = set(before), set(after)
    return {"right_to_wrong": len(a - b), "wrong_to_right": len(b - a)}
