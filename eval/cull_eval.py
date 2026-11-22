"""Culling labels on the dev half: which photos to show, cull_labels.csv, and the scores.

Rejects live in the tail, so the labeler sees the lowest-scoring quarter of the dev photos
(the union of the lowest of each score: the sharpest tiles, the whole photo's sharpness, the
brightness, and the highest blown-out share, grown until it holds a quarter) plus a random
``REST_SAMPLE`` of the others, shuffled together. Each photo from the rest stands for
(rest size / sample size) photos, which the recall estimate uses; precision is exact as long
as every flagged photo is in the labeled tail, and the report says when one is not.

A burst is shown when all its shots are dev rows; the labeler picks the shot to keep, or
says any is fine.
"""

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cull_page import CULL_COLUMNS, PHOTO_LABELS

CULL_CSV = "cull_labels.csv"
CULL_HTML = "cull.html"
REST_SAMPLE = 50
TAIL_SHARE = 0.25
BURST_SEPARATOR = " | "


@dataclass
class CullRow:
    kind: str  # 'photo' or 'burst'
    key: str  # a photo's file in labels.csv; a burst's shots' files, joined
    split: str
    sample: str  # photos: 'tail' or 'rest'
    weight: float
    label: str  # photos: '', or one of PHOTO_LABELS; bursts: '', 'any', or a shot's file


@dataclass(frozen=True)
class ScoredPhoto:
    key: str
    sharpness: float
    sharp_tiles: float
    luma_mean: float
    bright_frac: float


def read_cull_rows(path: Path) -> list[CullRow]:
    with path.open(newline="", encoding="utf-8") as fp:
        return [
            CullRow(
                kind=raw["kind"],
                key=raw["key"],
                split=raw["split"],
                sample=raw["sample"],
                weight=float(raw["weight"] or 0),
                label=(raw["label"] or "").strip(),
            )
            for raw in csv.DictReader(fp)
        ]


def write_cull_rows(path: Path, rows: list[CullRow]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CULL_COLUMNS)
        for r in rows:
            writer.writerow([r.kind, r.key, r.split, r.sample, r.weight, r.label])
    tmp.replace(path)


def pick_photos(photos: list[ScoredPhoto], seed: int = 0) -> list[tuple[ScoredPhoto, str, float]]:
    """(photo, 'tail' or 'rest', weight) for the photos to label, shuffled."""
    target = math.ceil(TAIL_SHARE * len(photos))
    orders = [
        sorted(photos, key=lambda p: p.sharp_tiles),
        sorted(photos, key=lambda p: p.sharpness),
        sorted(photos, key=lambda p: p.luma_mean),
        sorted(photos, key=lambda p: -p.bright_frac),
    ]
    tail: set[str] = set()
    k = 0
    while len(tail) < target and k < len(photos):
        k += 1
        tail = {p.key for order in orders for p in order[:k]}
    rest = [p for p in photos if p.key not in tail]
    rng = random.Random(seed)
    picked = rng.sample(rest, min(REST_SAMPLE, len(rest)))
    weight = len(rest) / len(picked) if picked else 0.0
    chosen = [(p, "tail", 1.0) for p in photos if p.key in tail] + [
        (p, "rest", weight) for p in picked
    ]
    rng.shuffle(chosen)
    return chosen


REVIEW_BURSTS = 40


def pick_review(
    flagged: list[str], others: list[str], bursts: list[str], seed: int = 0
) -> tuple[list[tuple[str, str, float]], list[str]]:
    """For a fresh trip: every flagged photo, a random ``REST_SAMPLE`` of the others (each
    standing for others / sample photos), shuffled; and a random ``REVIEW_BURSTS`` bursts."""
    rng = random.Random(seed)
    picked = rng.sample(others, min(REST_SAMPLE, len(others)))
    weight = len(others) / len(picked) if picked else 0.0
    chosen = [(key, "flagged", 1.0) for key in flagged] + [(key, "rest", weight) for key in picked]
    rng.shuffle(chosen)
    return chosen, sorted(rng.sample(bursts, min(REVIEW_BURSTS, len(bursts))))


def cull_report(
    rows: list[CullRow], reasons: dict[str, str], keepers: dict[str, str]
) -> dict[str, Any]:
    """``reasons``: photo file -> tripsort's reject reason, for the flagged dev photos;
    ``keepers``: a burst's key -> the file of the shot tripsort keeps."""
    photos = [r for r in rows if r.kind == "photo" and r.label in PHOTO_LABELS]
    flagged = [r for r in photos if r.key in reasons]
    bad = [r for r in photos if r.label != "ok"]
    weighted_bad = sum(r.weight for r in bad)
    weighted_found = sum(r.weight for r in bad if r.key in reasons)
    labeled_keys = {r.key for r in photos}
    per_reason = {}
    for reason in ("dark", "blown_out", "blurry"):
        mine = [r for r in flagged if reasons[r.key] == reason]
        per_reason[reason] = {
            "flagged": len(mine),
            "labeled_bad": sum(r.label != "ok" for r in mine),
            "labeled_this_reason": sum(r.label == reason for r in mine),
        }
    bursts = [r for r in rows if r.kind == "burst" and r.label]
    agree = sum(1 for r in bursts if r.label == "any" or r.label == keepers.get(r.key))
    return {
        "photos_labeled": len(photos),
        "labels": dict(Counter(r.label for r in photos)),
        "labels_weighted": {
            label: round(sum(r.weight for r in photos if r.label == label), 1)
            for label in PHOTO_LABELS
        },
        "flagged_labeled": len(flagged),
        "flagged_not_labeled": sum(1 for key in reasons if key not in labeled_keys),
        "precision": round(sum(r.label != "ok" for r in flagged) / len(flagged), 4)
        if flagged
        else None,
        "recall_estimate": round(weighted_found / weighted_bad, 4) if weighted_bad else None,
        "per_reason": per_reason,
        "bursts_labeled": len(bursts),
        "bursts_any": sum(1 for r in bursts if r.label == "any"),
        "burst_agreement": round(agree / len(bursts), 4) if bursts else None,
    }
