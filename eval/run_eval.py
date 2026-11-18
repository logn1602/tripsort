"""Evaluate tripsort. For now: dedupe precision and recall on synthetic copies.

    uv run tools/make_synthetic_copies.py DATA_DIR --generate 200
    uv run eval/run_eval.py dedupe DATA_DIR [--name generated] [--write-baseline]

Only numbers are printed and saved: no file names, no image data. ``--write-baseline``
stores them in eval/baseline.json under ``dedupe.<name>``.
"""

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (  # the sibling module, importable once the path is set
    Item,
    copies_left_in_all_by_variant,
    distance_histogram,
    false_merges_by_family,
    keeper_accuracy,
    outcome,
    pairwise,
    recall_by_variant,
    wrongly_left_out_by_family,
)

from tripsort import __version__
from tripsort.api import Project
from tripsort.ingest.hashing import hamming, phash_to_int

BASELINE = Path(__file__).resolve().parent / "baseline.json"


def evaluate_dedupe(data_dir: Path, work: Path) -> dict[str, Any]:
    with (data_dir / "truth.csv").open(encoding="utf-8", newline="") as fp:
        truth = list(csv.DictReader(fp))
    started = time.perf_counter()
    with Project.open(work) as project:
        project.ingest([data_dir / "originals.zip", data_dir / "copies.zip"])
        analysis = project.analyze()
        conn = project.connection
        media_of = {
            (label, member): (int(media_id), phash)
            for label, member, media_id, phash in conn.execute(
                "SELECT s.label, o.member_path, m.id, m.phash FROM occurrences o "
                "JOIN sources s ON s.id = o.source_id JOIN media m ON m.id = o.media_id"
            )
        }
        keeper_of = {
            int(media_id): int(keeper)
            for media_id, keeper in conn.execute(
                "SELECT gm.media_id, k.media_id FROM group_members gm "
                "JOIN media_groups g ON g.id = gm.group_id AND g.kind = 'copy' "
                "JOIN group_members k ON k.group_id = gm.group_id AND k.is_keeper = 1 "
                "WHERE gm.is_keeper = 0"
            )
        }
        thresholds = project.config.to_dict()["thresholds"]
    seconds = time.perf_counter() - started

    original_hash: dict[str, str | None] = {}
    for row in truth:
        if row["role"] == "original":
            original_hash[row["origin"]] = media_of[(row["zip"], row["member"])][1]

    items: list[Item] = []
    member_of_media: dict[int, str] = {}
    for row in truth:
        media_id, phash = media_of[(row["zip"], row["member"])]
        member_of_media.setdefault(media_id, row["member"])
    for row in truth:
        media_id, phash = media_of[(row["zip"], row["member"])]
        keeper = keeper_of.get(media_id, media_id)
        distance = None
        reference = original_hash.get(row["origin"])
        if row["role"] == "copy" and phash and reference:
            distance = hamming(phash_to_int(phash), phash_to_int(reference))
        items.append(
            Item(
                member=row["member"],
                origin=row["origin"],
                role=row["role"],
                variant=row["variant"],
                cluster=member_of_media[keeper],
                distance=distance,
            )
        )

    negatives = _negative_pair_distances(items, media_of, truth)
    roles = defaultdict(int)
    for item in items:
        roles[item.role] += 1
    return {
        "tripsort_version": __version__,
        "items": len(items),
        "originals": roles["original"],
        "copies": roles["copy"],
        "negatives": roles["negative"],
        "thresholds": thresholds,
        "dedupe": outcome(items),
        "wrongly_left_out_by_family": wrongly_left_out_by_family(items),
        "copies_left_in_all_by_variant": copies_left_in_all_by_variant(items),
        "keeper_accuracy": keeper_accuracy(items),
        "grouping": {
            "pairwise": pairwise(items),
            "grouped_with_own_original_by_variant": recall_by_variant(items),
            "grouped_with_another_photo_by_family": false_merges_by_family(items),
        },
        "copy_distance_histogram": distance_histogram(
            i.distance for i in items if i.role == "copy"
        ),
        "negative_pair_distance_histogram": negatives,
        "not_hashed": sum(1 for (_, phash) in media_of.values() if phash is None),
        "ingest_and_analyze_seconds": round(seconds, 1),
        "unique_files": analysis.stats.unique,
    }


def _negative_pair_distances(
    items: list[Item],
    media_of: dict[tuple[str, str], tuple[int, str | None]],
    truth: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    """pHash distances inside each negative family, pairing siblings made from one original."""
    phash_of = {row["member"]: media_of[(row["zip"], row["member"])][1] for row in truth}
    by_family: defaultdict[str, list[str]] = defaultdict(list)
    for item in items:
        if item.role == "negative":
            by_family[item.variant].append(item.member)
    result = {}
    for family, members in sorted(by_family.items()):
        distances = []
        for a, b in zip(members[::2], members[1::2], strict=False):  # consecutive siblings
            ha, hb = phash_of[a], phash_of[b]
            distances.append(hamming(phash_to_int(ha), phash_to_int(hb)) if ha and hb else None)
        result[family] = distance_histogram(distances)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dedupe = sub.add_parser("dedupe", help="precision and recall on synthetic copies")
    dedupe.add_argument("data", type=Path, help="folder from tools/make_synthetic_copies.py")
    dedupe.add_argument("--work", type=Path, help="project folder to use (default: a temp folder)")
    dedupe.add_argument("--name", default="generated", help="baseline key (default: generated)")
    dedupe.add_argument("--write-baseline", action="store_true", help="save to eval/baseline.json")
    args = parser.parse_args(argv)

    if args.work is not None:
        result = evaluate_dedupe(args.data, args.work)
    else:
        with tempfile.TemporaryDirectory(prefix="tripsort-eval-") as tmp:
            result = evaluate_dedupe(args.data, Path(tmp) / "project")
    print(json.dumps(result, indent=2))
    if args.write_baseline:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
        baseline.setdefault("dedupe", {})[args.name] = result
        BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"saved to {BASELINE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
