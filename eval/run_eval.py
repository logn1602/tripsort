"""Evaluate tripsort: dedupe on synthetic copies, people on the private labeled set.

    uv run tools/make_synthetic_copies.py DATA_DIR --generate 200
    uv run eval/run_eval.py dedupe DATA_DIR [--name generated] [--write-baseline]

    uv run eval/run_eval.py prelabel [--refs REFS] [--work DIR]
    uv run eval/run_eval.py people [--refs REFS] [--work DIR] [--split dev|test]
                                   [--set people.KEY=VALUE ...] [--name NAME] [--write-baseline]

The labeled set is the folder named by ``TRIPSORT_EVAL_DIR`` (see eval/datasets.py).
``prelabel`` writes labels.csv (unless it exists), filled with tripsort's guesses, and
labels.html, a click-to-label page for it, into that folder (see eval/labels_page.py);
``people`` scores the checked rows of labels.csv. With ``--refs``,
both also check that no ref photo is a copy of a labeled photo (see eval/ref_copies.py).
``--work`` keeps the zips and the index between runs, so a run after the first only analyzes.

Only numbers are printed and saved: no file names, no image data, and people appear as p1,
p2, ... ``--write-baseline`` stores the numbers in eval/baseline.json.
"""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib

from datasets import Source, build_sources, eval_dir, file_key  # the sibling modules
from labels import Row, person_key, read_rows, split_for, write_rows
from labels_page import THUMB_SIDE, Box, Card, thumb_uri, write_page
from metrics import (  # the sibling module, importable once the path is set
    Item,
    copies_left_in_all_by_variant,
    distance_histogram,
    false_merges_by_family,
    keeper_accuracy,
    merge_source_photos,
    outcome,
    pairwise,
    recall_by_variant,
    source_photo_merges,
    wrongly_left_out_by_family,
)
from people_metrics import (
    Photo,
    category_counts,
    face_report,
    people_report,
    shot_report,
    solo_face,
)
from PIL import Image
from ref_copies import ref_copies

from tripsort import __version__
from tripsort.api import Project
from tripsort.faces.names import display_name
from tripsort.faces.people import RefsResult
from tripsort.faces.types import FACE_MODEL_ID
from tripsort.index.db import transaction
from tripsort.ingest.decode import UnreadableImageError, decode_oriented
from tripsort.ingest.hashing import hamming, phash_to_int
from tripsort.ingest.sources import ZipCache

BASELINE = Path(__file__).resolve().parent / "baseline.json"
LABELS_CSV = "labels.csv"
LABELS_HTML = "labels.html"


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
        # When the sources are real photos, some may be real copies of each other; these
        # merges have no ground truth. "dedupe" counts each as a mistake (the lower bound);
        # this counts each as a correct merge (the upper bound).
        "source_photo_merges": source_photo_merges(items),
        "dedupe_if_source_merges_are_real_copies": outcome(merge_source_photos(items)),
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


@dataclass
class Prediction:
    """What tripsort decided about one kept photo of the labeled set."""

    media_id: int
    sha256: str
    people: list[str]  # names of the people whose folders it is in, person_NN included
    shot: str | None
    selfie: bool
    foreground: list[str | None] = field(default_factory=list)  # person of each fg face
    boxes: list[Box] = field(default_factory=list)
    # the people in ``people`` with a real name: the labels.csv guess, since only people
    # with a ref are labeled
    named: list[str] = field(default_factory=list)


def people_project(
    root: Path, work: Path, refs: Path | None, overrides: dict[str, str]
) -> tuple[Project, list[Source], RefsResult | None]:
    """Ingest the labeled set (once), then analyze from scratch: people from an earlier run
    are forgotten, so every run clusters like a fresh project would. The third value is how
    the refs went (None without refs); it names people and files, so never print it."""
    sources = build_sources(root, work / "zips")
    folder = work / "project"
    folder.mkdir(parents=True, exist_ok=True)
    _write_overrides(folder, overrides)
    project = Project.open(folder)
    project.ingest([s.zip_path for s in sources])
    with transaction(project.connection) as conn:
        conn.execute("UPDATE faces SET person_id = NULL, match = NULL, cluster = NULL")
        conn.execute("DELETE FROM people")
        conn.execute("DELETE FROM meta WHERE key IN ('next_person_number', 'analysis_state')")
        if refs is None:
            conn.execute("DELETE FROM ref_images")
    if refs is not None:
        return project, sources, project.people.name_from_refs(refs)
    project.analyze()
    return project, sources, None


def _refs_report(project: Project, refs: Path | None, named: RefsResult | None) -> dict[str, Any]:
    """How the refs went, in counts only (``named`` itself names people and files), and
    whether any ref photo is a copy of a labeled photo. Both None without refs."""
    if refs is None or named is None:
        return {"refs": None, "ref_copies": None}
    states = Counter(named.names.values())
    return {
        "refs": {
            "names": len(named.names),
            "named": states["named"],
            "no_match": states["no match"],
            "no_usable_photo": states["no usable photo"],
            "files_skipped": len(named.problems),
        },
        "ref_copies": ref_copies(
            project.connection,
            refs,
            FACE_MODEL_ID,
            project.config.thresholds.phash_max_distance,
        ),
    }


def _write_overrides(folder: Path, overrides: dict[str, str]) -> None:
    path = folder / "tripsort.toml"
    if not overrides:
        path.unlink(missing_ok=True)
        return
    tables: defaultdict[str, list[str]] = defaultdict(list)
    for dotted, value in sorted(overrides.items()):
        table, _, key = dotted.partition(".")
        tables[table].append(f"{key} = {value}")
    path.write_text(
        "".join(f"[{t}]\n" + "\n".join(lines) + "\n" for t, lines in tables.items()),
        encoding="utf-8",
    )


def predictions(
    project: Project, sources: list[Source]
) -> tuple[dict[str, int], dict[int, Prediction]]:
    """(labels.csv file -> media id for every occurrence, media id -> prediction for every
    kept photo)."""
    conn = project.connection
    by_label = {s.label: s for s in sources}
    files = {
        file_key(by_label[str(label)], str(member)): int(media_id)
        for media_id, label, member in conn.execute(
            "SELECT o.media_id, s.label, o.member_path FROM occurrences o "
            "JOIN sources s ON s.id = o.source_id"
        )
        if str(label) in by_label
    }
    kept: dict[int, Prediction] = {}
    for media_id, sha, shot, selfie in conn.execute(
        "SELECT id, sha256, shot, is_selfie FROM media WHERE kind = 'image' AND status = 'ok' "
        "AND id IN (SELECT media_id FROM occurrences) AND id NOT IN ("
        "  SELECT gm.media_id FROM group_members gm JOIN media_groups g ON g.id = gm.group_id "
        "  WHERE g.kind = 'copy' AND gm.is_keeper = 0)"
    ):
        kept[int(media_id)] = Prediction(int(media_id), str(sha), [], shot, bool(selfie))
    min_score = project.config.people.min_face_score
    for media_id, x, y, w, h, fg, score, match, number, name in conn.execute(
        "SELECT f.media_id, f.x, f.y, f.w, f.h, f.is_foreground, f.score, f.match, p.number, "
        "p.name FROM faces f LEFT JOIN people p ON p.id = f.person_id "
        "WHERE f.model_id = ? ORDER BY f.media_id, f.face_index",
        (FACE_MODEL_ID,),
    ):
        prediction = kept.get(int(media_id))
        if prediction is None or score < min_score:
            continue
        person = display_name(int(number), name) if match is not None else None
        if person is not None and person not in prediction.people:
            prediction.people.append(person)
            if name is not None:
                prediction.named.append(person)
        if fg:
            prediction.foreground.append(person)
        prediction.boxes.append(Box(x, y, w, h, person or "?", bool(fg)))
    return files, kept


def _guess_rows(files: dict[str, int], kept: dict[int, Prediction]) -> list[Row]:
    first_file = {}
    for name, media_id in sorted(files.items()):
        first_file.setdefault(media_id, name)
    ordered = sorted(kept.values(), key=lambda p: p.sha256)  # a random-looking, stable order
    return [
        Row(
            row=number,
            file=first_file[p.media_id],
            split=split_for(p.sha256),
            checked=False,
            people=sorted(p.named),
            shot=p.shot or "",
            selfie=p.selfie,
            category="other" if p.shot in ("solo", "group") else "",
        )
        for number, p in enumerate(ordered, start=1)
    ]


def prelabel(
    root: Path,
    work: Path,
    refs: Path | None,
    overrides: dict[str, str] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write labels.csv (unless it exists) and labels.html into the labeled set's folder.
    The page starts from labels.csv as it is, with tripsort's guesses as the face boxes."""
    project, sources, named = people_project(root, work, refs, overrides or {})
    with project:
        refs_report = _refs_report(project, refs, named)
        files, kept = predictions(project, sources)
        csv_path = root / LABELS_CSV
        if csv_path.exists() and not force:
            rows, problems = read_rows(csv_path)
            state = "kept as it was"
        else:
            rows, problems = _guess_rows(files, kept), []
            write_rows(csv_path, rows)
            state = "written"
        csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        names = _button_names(named, rows)
        cards = []
        missing = 0
        with ZipCache(project.connection) as zips:
            for row in rows:
                card = Card(
                    row=row.row,
                    file=row.file,
                    split=row.split,
                    checked=row.checked,
                    people=list(row.people),
                    shot=row.shot,
                    selfie=row.selfie,
                    category=row.category,
                )
                media_id = files.get(row.file)
                guess = kept.get(media_id) if media_id is not None else None
                if guess is not None:
                    card.boxes = guess.boxes
                    data = zips.read(guess.media_id, guess.sha256)
                    if data is not None:
                        with contextlib.suppress(UnreadableImageError):
                            pixels = decode_oriented(data, max_side=THUMB_SIDE * 2)
                            card.thumb = thumb_uri(Image.fromarray(pixels))
                if card.thumb is None:
                    missing += 1
                cards.append(card)
        # saving from the page would drop the rows labels.csv could not read
        write_page(root / LABELS_HTML, cards, names, csv_sha256=csv_sha256, can_save=not problems)
        people_found = {name for p in kept.values() for name in p.people}
        named_found = {name for p in kept.values() for name in p.named}
    return {
        "rows": len(rows),
        "dev": sum(r.split == "dev" for r in rows),
        "test": sum(r.split == "test" for r in rows),
        "checked": sum(r.checked for r in rows),
        "label_problems": len(problems),
        "rows_without_photo": missing,
        **refs_report,
        "people_found": len(people_found),
        "named_people": len(named_found),
        "photos_with_people": sum(bool(p.people) for p in kept.values()),
        "photos_with_named_people": sum(bool(p.named) for p in kept.values()),
        "labels_csv": state,
        "labels_html": "written",
        "name_buttons": len(names),
    }


def _button_names(named: RefsResult | None, rows: list[Row]) -> list[str]:
    """The names labels.html has a button for: every ref name, or without refs, every name in
    labels.csv (its first spelling). Ignoring case, alphabetical."""
    if named is not None:
        names = list(named.names)
    else:
        first: dict[str, str] = {}
        for row in rows:
            for name in row.people:
                first.setdefault(person_key(name), name)
        names = list(first.values())
    return sorted(names, key=lambda n: (n.casefold(), n))


def evaluate_people(
    root: Path, work: Path, refs: Path | None, split: str, overrides: dict[str, str]
) -> dict[str, Any]:
    started = time.perf_counter()
    rows, problems = read_rows(root / LABELS_CSV)
    if problems:  # row numbers and column names only
        return {"label_problems": len(problems), "first_problems": problems[:20]}
    project, sources, named = people_project(root, work, refs, overrides)
    with project:
        refs_report = _refs_report(project, refs, named)
        files, kept = predictions(project, sources)
        settings = asdict(project.config.people)
    photos = []
    not_found = 0
    # The labeled people: anyone named in a checked row, of either split, and every ref name,
    # even one no checked row lists (the labeler lists them in every row they are in).
    everyone: Counter[str] = Counter(
        key for r in rows if r.checked for key in {person_key(n) for n in r.people}
    )
    ref_keys = {person_key(name) for name in named.names} if named is not None else set()
    # with refs, a name in the rows that matches no ref is most likely a typo
    without_ref = len(set(everyone) - ref_keys) if named is not None else None
    for key in ref_keys:
        everyone.setdefault(key, 0)
    checked = [r for r in rows if r.checked and r.split == split]
    for row in checked:
        media_id = files.get(row.file)
        guess = kept.get(media_id) if media_id is not None else None
        if guess is None:
            not_found += 1
            continue
        labeled = {person_key(n) for n in row.people}
        foreground = [person_key(f) if f else None for f in guess.foreground]
        photos.append(
            Photo(
                labeled=labeled,
                predicted={person_key(n) for n in guess.people},
                shot=row.shot,
                predicted_shot=guess.shot,
                selfie=row.selfie,
                predicted_selfie=guess.selfie,
                solo_face=solo_face(labeled, row.shot, foreground),
                category=row.category,
            )
        )
    return {
        "tripsort_version": __version__,
        "split": split,
        "checked_rows": len(checked),
        "rows_not_found": not_found,
        **refs_report,
        "labeled_names_without_a_ref": without_ref,
        "settings": settings,
        "people": people_report(photos, everyone),
        "shots": shot_report(photos),
        "faces": face_report(photos),
        "categories_labeled": category_counts(photos),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _hidden_errors(run: Callable[[], dict[str, Any]]) -> dict[str, Any] | None:
    """Run ``run``; on an error print its type and code locations, never its message (it may
    name a photo or a person)."""
    try:
        return run()
    except Exception as exc:
        frames = [
            f"{Path(f.filename).name}:{f.lineno} in {f.name}"
            for f in traceback.extract_tb(exc.__traceback__)
        ]
        print(
            f"failed: {type(exc).__name__} (message hidden)\n  " + "\n  ".join(frames),
            file=sys.stderr,
        )
        return None


def _save_baseline(section: str, name: str, result: dict[str, Any]) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    baseline.setdefault(section, {})[name] = result
    BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved to {BASELINE}", file=sys.stderr)


def _with_work(work: Path | None, run: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    if work is not None:
        return run(work)
    with tempfile.TemporaryDirectory(prefix="tripsort-eval-") as tmp:
        return run(Path(tmp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dedupe = sub.add_parser("dedupe", help="precision and recall on synthetic copies")
    dedupe.add_argument("data", type=Path, help="folder from tools/make_synthetic_copies.py")
    dedupe.add_argument("--work", type=Path, help="project folder to use (default: a temp folder)")
    dedupe.add_argument("--name", default="generated", help="baseline key (default: generated)")
    dedupe.add_argument("--write-baseline", action="store_true", help="save to eval/baseline.json")
    for name, text in (
        ("prelabel", "write labels.csv and labels.html into the labeled set's folder"),
        ("people", "per-person precision and recall on the checked rows of labels.csv"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--refs", type=Path, help="refs folder: one photo per friend")
        p.add_argument("--work", type=Path, help="keep zips and index here between runs")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="TABLE.KEY=VALUE",
            help="a setting to use, such as people.min_cluster_size=4",
        )
        if name == "prelabel":
            p.add_argument("--force", action="store_true", help="replace an existing labels.csv")
        else:
            p.add_argument("--split", choices=("dev", "test"), default="dev")
            p.add_argument("--name", default="private_dev", help="baseline key")
            p.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "dedupe":
        result = _with_work(args.work, lambda w: evaluate_dedupe(args.data, w / "project"))
        print(json.dumps(result, indent=2))
        if args.write_baseline:
            _save_baseline("dedupe", args.name, result)
        return 0

    root = eval_dir()
    overrides = dict(item.split("=", 1) for item in args.set)
    if args.command == "prelabel":
        outcome = _hidden_errors(
            lambda: _with_work(
                args.work,
                lambda w: prelabel(root, w, args.refs, overrides, force=args.force),
            )
        )
    else:
        outcome = _hidden_errors(
            lambda: _with_work(
                args.work, lambda w: evaluate_people(root, w, args.refs, args.split, overrides)
            )
        )
    if outcome is None:
        return 1
    print(json.dumps(outcome, indent=2))
    if args.command == "people" and args.write_baseline and "people" in outcome:
        _save_baseline("people", args.name, outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
