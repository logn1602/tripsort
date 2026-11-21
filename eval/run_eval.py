"""Evaluate tripsort: dedupe on synthetic copies, people on the private labeled set.

    uv run tools/make_synthetic_copies.py DATA_DIR --generate 200
    uv run eval/run_eval.py dedupe DATA_DIR [--name generated] [--write-baseline]

    uv run eval/run_eval.py prelabel [--refs REFS] [--work DIR] [--categories-only]
    uv run eval/run_eval.py people [--refs REFS] [--work DIR] [--split dev|test]
                                   [--set people.KEY=VALUE ...] [--name NAME] [--write-baseline]
    uv run eval/run_eval.py categories [--work DIR] [--split dev|test] [--config FILE]
                                       [--mistakes] [--name NAME] [--write-baseline]

Each takes --labeled-set DIR for a set other than TRIPSORT_EVAL_DIR (such as a category-only
set), --config FILE (a tripsort.toml to use, such as a prompt set), and --set TABLE.KEY=VALUE.

The labeled set is the folder named by ``TRIPSORT_EVAL_DIR`` (see eval/datasets.py).
``prelabel`` writes labels.csv (unless it exists), filled with tripsort's guesses, and
labels.html, a click-to-label page for it, into that folder (see eval/labels_page.py);
``people`` scores the checked rows of labels.csv, and the merge suggestions: how many the
dev rows confirm, and the scores after accepting the confirmed ones or all of them. With
``--refs``, both also check that no ref photo is a copy of a labeled photo (see
eval/ref_copies.py).
``--work`` keeps the zips and the index between runs, so a run after the first only analyzes.
``categories`` scores the primary category of the checked rows (precision, recall, and F1
per category, the confusion matrix) and the selfie hint; with ``--mistakes`` it writes
category_mistakes.html, the dev rows it got wrong, into the labeled set's folder.

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
import tomllib
import traceback
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib
import html
import itertools

import numpy as np
from category_metrics import category_report
from datasets import Source, build_sources, eval_dir, file_key  # the sibling modules
from labels import CATEGORIES, Row, person_key, read_rows, split_for, write_rows
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
    aliases,
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
from tripsort.clip.types import CLIP_MODEL_ID
from tripsort.faces.names import display_name
from tripsort.faces.people import RefsResult
from tripsort.faces.suggest import Suggestion
from tripsort.faces.types import FACE_MODEL_ID
from tripsort.index.db import transaction
from tripsort.ingest.decode import UnreadableImageError, decode_oriented
from tripsort.ingest.hashing import hamming, phash_to_int
from tripsort.ingest.sources import ZipCache
from tripsort.ingest.zip_reader import IMAGE_EXTENSIONS

BASELINE = Path(__file__).resolve().parent / "baseline.json"
LABELS_CSV = "labels.csv"
LABELS_HTML = "labels.html"
MISTAKES_HTML = "category_mistakes.html"


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
    category: str | None = None  # the primary category, None when not scored


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
    for media_id, label in conn.execute(
        "SELECT media_id, label FROM categories WHERE is_primary = 1"
    ):
        if (prediction := kept.get(int(media_id))) is not None:
            prediction.category = str(label)
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


def _guess_rows(
    files: dict[str, int], kept: dict[int, Prediction], *, categories_only: bool = False
) -> list[Row]:
    """labels.csv rows pre-filled with tripsort's guesses. For a category-only set (photos
    with no people) the shot is none and only the category is guessed."""
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
            people=[] if categories_only else sorted(p.named),
            shot="none" if categories_only else p.shot or "",
            selfie=False if categories_only else p.selfie,
            category=p.category or ("other" if p.shot in ("solo", "group") else ""),
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
    categories_only: bool = False,
) -> dict[str, Any]:
    """Write labels.csv (unless it exists) and labels.html into the labeled set's folder.
    The page starts from labels.csv as it is, with tripsort's guesses as the face boxes.
    ``categories_only`` is for a set with no people: the page offers the category alone."""
    project, sources, named = people_project(root, work, refs, overrides or {})
    with project:
        refs_report = _refs_report(project, refs, named)
        files, kept = predictions(project, sources)
        csv_path = root / LABELS_CSV
        if csv_path.exists() and not force:
            rows, problems = read_rows(csv_path)
            state = "kept as it was"
        else:
            rows, problems = _guess_rows(files, kept, categories_only=categories_only), []
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
        write_page(
            root / LABELS_HTML,
            cards,
            names,
            csv_sha256=csv_sha256,
            can_save=not problems,
            categories_only=categories_only,
        )
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
        suggestions = project.people.suggestions()
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
    photos, checked, not_found = _scored_photos(rows, split, files, kept)
    return {
        "tripsort_version": __version__,
        "split": split,
        "checked_rows": checked,
        "rows_not_found": not_found,
        **refs_report,
        "labeled_names_without_a_ref": without_ref,
        "settings": settings,
        "people": people_report(photos, everyone),
        "shots": shot_report(photos),
        "faces": face_report(photos),
        "categories_labeled": category_counts(photos),
        "suggestions": evaluate_suggestions(
            (root, work, refs, overrides), split, rows, files, kept, suggestions, everyone
        ),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _scored_photos(
    rows: list[Row], split: str, files: dict[str, int], kept: dict[int, Prediction]
) -> tuple[list[Photo], int, int]:
    """The checked rows of ``split`` with tripsort's guesses: (photos, checked rows, rows
    whose photo was not found)."""
    photos = []
    not_found = 0
    checked = [r for r in rows if r.checked and r.split == split]
    for row in checked:
        guess = _guess(row, files, kept)
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
    return photos, len(checked), not_found


def _guess(row: Row, files: dict[str, int], kept: dict[int, Prediction]) -> Prediction | None:
    media_id = files.get(row.file)
    return kept.get(media_id) if media_id is not None else None


# (labeled set folder, work folder, refs folder, settings): what it takes to start over
Setup = tuple[Path, Path, Path | None, dict[str, str]]


def evaluate_suggestions(
    setup: Setup,
    split: str,
    rows: list[Row],
    files: dict[str, int],
    kept: dict[int, Prediction],
    suggestions: list[Suggestion],
    everyone: Counter[str],
) -> dict[str, Any]:
    """How the merge suggestions fare. For each candidate, the dev photos of the person
    suggested that list the named person (right) or not (wrong). These always come from the
    dev rows, whichever split is scored, so the choice of what to accept never looks at the
    test half. A candidate is confirmed when it has right photos and no wrong one.

    Then the split's scores after merging, from a fresh clustering each time: the confirmed
    candidates (a user who checks every suggestion and is always right), and every
    suggestion's first candidate (a user who accepts them all without looking)."""
    alias = aliases(everyone)
    dev = [(r, g) for r in rows if r.checked and r.split == "dev" if (g := _guess(r, files, kept))]
    listed: list[dict[str, Any]] = []
    confirmed: list[tuple[str, str]] = []
    blind: list[tuple[str, str]] = []
    for suggestion in suggestions:
        own = [r for r, guess in dev if suggestion.person in guess.people]
        chosen = None
        for rank, candidate in enumerate(suggestion.candidates, 1):
            key = person_key(candidate.name)
            right = sum(1 for r in own if key in {person_key(n) for n in r.people})
            ok = right > 0 and right == len(own)
            listed.append(
                {
                    "named": alias.get(key, "not labeled"),
                    "rank": rank,
                    "similarity": round(candidate.similarity, 3),
                    "photos": suggestion.photos,
                    "dev_photos": len(own),
                    "right": right,
                    "wrong": len(own) - right,
                    "confirmed": ok,
                }
            )
            if ok and chosen is None:
                chosen = candidate.name
        if chosen is not None:
            confirmed.append((suggestion.person, chosen))
        blind.append((suggestion.person, suggestion.candidates[0].name))
    report: dict[str, Any] = {
        "shown": len(suggestions),
        "with_two_candidates": sum(len(s.candidates) > 1 for s in suggestions),
        "confirmed": len(confirmed),
        "candidates": listed,
    }
    if suggestions:
        for name, merges in (("confirmed", confirmed), ("all", blind)):
            report[f"after_accepting_{name}"] = _after_merging(setup, split, rows, everyone, merges)
    return report


def _after_merging(
    setup: Setup,
    split: str,
    rows: list[Row],
    everyone: Counter[str],
    merges: list[tuple[str, str]],
) -> dict[str, Any]:
    """The scores of ``split`` after merging each (person_NN, name) pair into the named
    person, starting again from the clustering a fresh project gets."""
    project, sources, _ = people_project(*setup)
    with project:
        shown = {(s.person, c.name) for s in project.people.suggestions() for c in s.candidates}
        merged = 0
        for person, name in merges:
            if (person, name) in shown:  # the same clustering gives the same suggestions
                project.people.merge(person, name)
                merged += 1
        project.analyze()
        files, kept = predictions(project, sources)
    photos, _, _ = _scored_photos(rows, split, files, kept)
    people = people_report(photos, everyone)
    return {
        "merged": merged,
        "micro": people["micro"],
        "macro": people["macro"],
        "per_person": people["per_person"],
        "faces": face_report(photos),
    }


def evaluate_categories(
    root: Path,
    work: Path,
    refs: Path | None,
    split: str,
    overrides: dict[str, str],
    *,
    mistakes: bool = False,
) -> dict[str, Any]:
    """The primary category of each checked row of ``split`` against its label, and the
    selfie hint: on the labeled selfies, and on free labels (photos whose lens model says
    which camera took them, scored by the hint alone)."""
    started = time.perf_counter()
    rows, problems = read_rows(root / LABELS_CSV)
    if problems:  # row numbers and column names only
        return {"label_problems": len(problems), "first_problems": problems[:20]}
    project, sources, _ = people_project(root, work, refs, overrides)
    with project:
        conn = project.connection
        files, kept = predictions(project, sources)
        config = project.config
        hint_photos = {
            int(media_id)
            for (media_id,) in conn.execute(
                "SELECT id FROM media WHERE is_front_camera IS NULL AND shot IN ('solo', 'group')"
            )
        }
        free = conn.execute(
            "SELECT is_front_camera, clip_selfie FROM media WHERE shot IN ('solo', 'group') "
            "AND is_front_camera IS NOT NULL AND clip_selfie IS NOT NULL"
        ).fetchall()
        page = _write_mistakes(root, project, rows, files, kept) if mistakes else None
        survey = burst_survey(conn)
    checked = [r for r in rows if r.checked and r.split == split]
    pairs: list[tuple[str, str]] = []
    right_rows: list[int] = []
    not_found = not_scored = 0
    tp = fp = fn = hint_used = hint_right = 0
    for row in checked:
        guess = _guess(row, files, kept)
        if guess is None:
            not_found += 1
            continue
        tp += row.selfie and guess.selfie
        fp += guess.selfie and not row.selfie
        fn += row.selfie and not guess.selfie
        if guess.media_id in hint_photos and guess.selfie:
            hint_used += 1
            hint_right += row.selfie
        if guess.category is None:
            not_scored += 1
            continue
        pairs.append((row.category, guess.category))
        if row.category == guess.category:
            right_rows.append(row.row)
    threshold = config.people.selfie_min_score
    free_tp = sum(1 for front, score in free if front and score >= threshold)
    free_fp = sum(1 for front, score in free if not front and score >= threshold)
    free_fn = sum(1 for front, score in free if front and score < threshold)
    return {
        "tripsort_version": __version__,
        "clip_model": CLIP_MODEL_ID,
        "split": split,
        "checked_rows": len(checked),
        "rows_not_found": not_found,
        "rows_not_scored": not_scored,
        "settings": {
            "category_min_score": config.thresholds.category_min_score,
            "categories": {name: list(p) for name, p in config.categories.items()},
            "selfie_min_score": threshold,
        },
        "categories": category_report(pairs, list(CATEGORIES)),
        "right_rows": sorted(right_rows),
        "selfie": {
            "labeled": tp + fn,
            "precision": _share(tp, tp + fp),
            "recall": _share(tp, tp + fn),
            "from_the_hint": hint_used,
            "from_the_hint_labeled_selfie": hint_right,
            "free_labels": {
                "photos": len(free),
                "front_camera": sum(1 for front, _ in free if front),
                "precision": _share(free_tp, free_tp + free_fp),
                "recall": _share(free_tp, free_tp + free_fn),
            },
        },
        "mistakes_page": page,
        "bursts_on_this_set": survey,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _share(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


_SIMILARITY_EDGES = (0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98)


def similarity_histogram(values: list[float]) -> dict[str, int]:
    """How many CLIP similarities fall in each band, the bands around burst_similarity."""
    labels = [f"<{_SIMILARITY_EDGES[0]:.2f}"]
    labels += [f"{a:.2f}-{b:.2f}" for a, b in itertools.pairwise(_SIMILARITY_EDGES)]
    labels.append(f">={_SIMILARITY_EDGES[-1]:.2f}")
    counts = dict.fromkeys(labels, 0)
    for value in values:
        band = sum(1 for edge in _SIMILARITY_EDGES if value >= edge)
        counts[labels[band]] += 1
    return counts


def burst_survey(conn: Any) -> dict[str, Any]:
    """On a real set, no burst labels: how alike consecutive shots of one camera are, by the
    time between them, and how many bursts the current settings find."""
    from tripsort.clip.bursts import load_shots

    by_camera: defaultdict[str, list[Any]] = defaultdict(list)
    for shot in load_shots(conn):
        by_camera[shot.camera].append(shot)
    gaps: dict[str, list[float]] = {"under_10s": [], "10_to_60s": [], "60_to_600s": []}
    for shots in by_camera.values():
        shots.sort(key=lambda s: (s.seconds, s.media_id))
        for a, b in itertools.pairwise(shots):
            gap = b.seconds - a.seconds
            band = "under_10s" if gap <= 10 else "10_to_60s" if gap <= 60 else "60_to_600s"
            if gap <= 600:
                gaps[band].append(float(a.embedding @ b.embedding))
    (bursts, photos) = conn.execute(
        "SELECT COUNT(DISTINCT g.id), COUNT(gm.media_id) FROM media_groups g "
        "JOIN group_members gm ON gm.group_id = g.id WHERE g.kind = 'burst'"
    ).fetchone()
    return {
        "bursts": int(bursts),
        "photos_in_bursts": int(photos),
        "consecutive_shots": {band: similarity_histogram(v) for band, v in gaps.items()},
    }


def evaluate_bursts(data_dir: Path, work: Path, overrides: dict[str, str]) -> dict[str, Any]:
    """Burst grouping on tools/make_synthetic_copies.py --bursts output: pairwise precision
    and recall against the true moments, burst shots missed, neighbors that joined a burst,
    and how alike CLIP finds a moment's shots, and a neighbor, to its first shot."""
    started = time.perf_counter()
    with (data_dir / "bursts_truth.csv").open(encoding="utf-8", newline="") as fp:
        truth = {r["member"]: (r["moment"], r["role"]) for r in csv.DictReader(fp)}
    folder = work / "project"
    folder.mkdir(parents=True, exist_ok=True)
    _write_overrides(folder, overrides)
    with Project.open(folder) as project:
        project.ingest([data_dir / "bursts.zip"])
        project.analyze()
        conn = project.connection
        media_of = {
            str(member): int(media_id)
            for media_id, member in conn.execute("SELECT media_id, member_path FROM occurrences")
        }
        group_of = {
            int(media_id): int(group_id)
            for group_id, media_id in conn.execute(
                "SELECT gm.group_id, gm.media_id FROM group_members gm "
                "JOIN media_groups g ON g.id = gm.group_id WHERE g.kind = 'burst'"
            )
        }
        vectors = {
            int(media_id): np.frombuffer(blob, dtype=np.float32)
            for media_id, blob in conn.execute(
                "SELECT media_id, embedding FROM clip_embeddings WHERE model_id = ?",
                (CLIP_MODEL_ID,),
            )
        }
        settings = {
            "burst_similarity": project.config.thresholds.burst_similarity,
            "burst_window_seconds": project.config.thresholds.burst_window_seconds,
        }
    members = sorted(m for m in truth if m in media_of)
    tp = fp = fn = 0
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            same = truth[a][0] == truth[b][0]
            ga, gb = group_of.get(media_of[a]), group_of.get(media_of[b])
            grouped = ga is not None and ga == gb
            tp += same and grouped
            fp += grouped and not same
            fn += same and not grouped
    firsts = {truth[m][0]: m for m in members if truth[m][1] == "first"}
    burst_sims, neighbor_sims = [], []
    missed = joined = 0
    for member in members:
        moment, role = truth[member]
        index = moment[1:]
        first = firsts.get(moment if role != "neighbor" else f"m{index}")
        if first is None or member == first:
            continue
        a, b = media_of[first], media_of[member]
        similarity = float(vectors[a] @ vectors[b]) if a in vectors and b in vectors else None
        same_group = group_of.get(a) is not None and group_of.get(a) == group_of.get(b)
        if role == "burst":
            missed += not same_group
            if similarity is not None:
                burst_sims.append(similarity)
        elif role == "neighbor":
            joined += same_group
            if similarity is not None:
                neighbor_sims.append(similarity)
    return {
        "tripsort_version": __version__,
        "clip_model": CLIP_MODEL_ID,
        "settings": settings,
        "shots": len(members),
        "moments": len({truth[m][0] for m in members}),
        "pairs": {
            "precision": _share(tp, tp + fp) if tp + fp else 1.0,
            "recall": _share(tp, tp + fn),
        },
        "burst_shots_missed": missed,
        "neighbors_joined": joined,
        "similarity_to_first_shot": {
            "burst_shots": similarity_histogram(burst_sims),
            "neighbors": similarity_histogram(neighbor_sims),
        },
        "seconds": round(time.perf_counter() - started, 1),
    }


_MISTAKES_STYLE = (
    "body{font:15px/1.4 system-ui,sans-serif;margin:0;padding:16px;background:#f7f7f5}"
    ".grid{display:flex;flex-wrap:wrap;gap:12px}"
    "figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px;"
    "width:260px}img{width:100%}figcaption{font-size:13px}"
    "@media (prefers-color-scheme:dark){body{background:#1b1b18;color:#eee}"
    "figure{background:#2a2a26;border-color:#444}}"
)


def _write_mistakes(
    root: Path,
    project: Project,
    rows: list[Row],
    files: dict[str, int],
    kept: dict[int, Prediction],
) -> dict[str, int]:
    """category_mistakes.html in the labeled set's folder: the checked dev rows whose primary
    category is not their label, with the photo and its top shares. Only the labeler opens
    it; what this returns is a count."""
    shares: defaultdict[int, dict[str, float]] = defaultdict(dict)
    for media_id, label, score in project.connection.execute(
        "SELECT media_id, label, score FROM categories"
    ):
        shares[int(media_id)][str(label)] = float(score)
    cards = []
    with ZipCache(project.connection) as zips:
        for row in rows:
            if not (row.checked and row.split == "dev"):
                continue
            guess = _guess(row, files, kept)
            if guess is None or guess.category is None or guess.category == row.category:
                continue
            thumb = ""
            data = zips.read(guess.media_id, guess.sha256)
            if data is not None:
                with contextlib.suppress(UnreadableImageError):
                    pixels = decode_oriented(data, max_side=THUMB_SIDE * 2)
                    thumb = thumb_uri(Image.fromarray(pixels))
            top = sorted(shares[guess.media_id].items(), key=lambda kv: -kv[1])[:3]
            cards.append((row.row, row.category, guess.category, top, thumb))
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Category mistakes</title><style>{_MISTAKES_STYLE}</style></head><body>",
        f"<h1>Category mistakes on the dev rows: {len(cards)}</h1>",
        "<p>Each photo with its labeled category and tripsort's, and tripsort's top shares. ",
        "This page shows your photos: keep it private.</p><div class='grid'>",
    ]
    for number, labeled, predicted, top, thumb in cards:
        shown = ", ".join(f"{label} {score:.2f}" for label, score in top)
        image = f"<img src='{thumb}' alt=''>" if thumb else "<p>photo not found</p>"
        parts.append(
            f"<figure>{image}<figcaption>row {number}: labeled <b>{html.escape(labeled)}</b>, "
            f"tripsort <b>{html.escape(predicted)}</b><br>{html.escape(shown)}</figcaption>"
            "</figure>"
        )
    parts.append("</div></body></html>")
    target = root / MISTAKES_HTML
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(parts) + "\n", encoding="utf-8")
    tmp.replace(target)
    return {"rows": len(cards)}


PUBLIC_SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "categories"


def evaluate_public(sample: Path, work: Path) -> dict[str, Any]:
    """The categories of a public sample (one folder per category of public-domain or CC0
    photos, such as tests/fixtures/categories), through the whole pipeline with the real
    models. A sanity check: the photos are clean web photos, not trip photos."""
    started = time.perf_counter()
    files = sorted(p for p in sample.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "public.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in files:
            zf.write(path, f"{path.parent.name}/{path.name}")
    with Project.open(work / "project") as project:
        project.ingest([archive])
        project.analyze()
        found = dict(
            project.connection.execute(
                "SELECT o.member_path, c.label FROM occurrences o JOIN categories c "
                "ON c.media_id = o.media_id AND c.is_primary = 1"
            ).fetchall()
        )
    pairs = [(m.split("/")[0], str(label)) for m, label in sorted(found.items())]
    return {
        "tripsort_version": __version__,
        "clip_model": CLIP_MODEL_ID,
        "photos": len(files),
        "scored": len(pairs),
        "categories": category_report(pairs, list(CATEGORIES)),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _config_overrides(path: Path) -> dict[str, str]:
    """A tripsort.toml as TABLE.KEY overrides (the key dotted further for nested tables,
    such as categories.food.prompts), so --set can still change single values."""
    out: dict[str, str] = {}

    def walk(keys: list[str], value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk([*keys, key], inner)
        else:
            out[".".join(_toml_key(k) for k in keys)] = _toml_value(value)

    walk([], tomllib.loads(path.read_text(encoding="utf-8")))
    return out


def _toml_key(key: str) -> str:
    bare = key.isascii() and key.replace("_", "").replace("-", "").isalnum()
    return key if bare else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return repr(value)


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
    bursts = sub.add_parser("bursts", help="burst grouping on make_synthetic_copies.py --bursts")
    bursts.add_argument("data", type=Path, help="folder from make_synthetic_copies.py --bursts")
    bursts.add_argument("--work", type=Path, help="project folder to use (default: a temp folder)")
    bursts.add_argument("--set", action="append", default=[], metavar="TABLE.KEY=VALUE")
    bursts.add_argument("--name", default="generated", help="baseline key (default: generated)")
    bursts.add_argument("--write-baseline", action="store_true", help="save to eval/baseline.json")
    public = sub.add_parser("public", help="categories of a public sample, with the real models")
    public.add_argument("sample", type=Path, nargs="?", default=PUBLIC_SAMPLE)
    public.add_argument("--work", type=Path, help="project folder to use (default: a temp folder)")
    public.add_argument("--name", default="public_sample", help="baseline key")
    public.add_argument("--write-baseline", action="store_true", help="save to eval/baseline.json")
    for name, text in (
        ("prelabel", "write labels.csv and labels.html into the labeled set's folder"),
        ("people", "per-person precision and recall on the checked rows of labels.csv"),
        ("categories", "per-category precision and recall, and the selfie hint"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--refs", type=Path, help="refs folder: one photo per friend")
        p.add_argument("--work", type=Path, help="keep zips and index here between runs")
        p.add_argument("--labeled-set", type=Path, help="default: the TRIPSORT_EVAL_DIR folder")
        p.add_argument("--config", type=Path, help="a tripsort.toml to use, such as prompts")
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="TABLE.KEY=VALUE",
            help="a setting to use, such as people.min_cluster_size=4",
        )
        if name == "prelabel":
            p.add_argument("--force", action="store_true", help="replace an existing labels.csv")
            p.add_argument("--categories-only", action="store_true", help="a set with no people")
        else:
            default_name = "private_dev" if name == "people" else "private_categories_dev"
            p.add_argument("--split", choices=("dev", "test"), default="dev")
            p.add_argument("--name", default=default_name, help="baseline key")
            p.add_argument("--write-baseline", action="store_true")
        if name == "categories":
            p.add_argument("--mistakes", action="store_true", help=f"write {MISTAKES_HTML}")
    args = parser.parse_args(argv)

    if args.command == "dedupe":
        result = _with_work(args.work, lambda w: evaluate_dedupe(args.data, w / "project"))
        print(json.dumps(result, indent=2))
        if args.write_baseline:
            _save_baseline("dedupe", args.name, result)
        return 0
    if args.command == "public":
        result = _with_work(args.work, lambda w: evaluate_public(args.sample, w))
        print(json.dumps(result, indent=2))
        if args.write_baseline:
            _save_baseline("categories", args.name, result)
        return 0
    if args.command == "bursts":
        burst_overrides = dict(item.split("=", 1) for item in args.set)
        result = _with_work(args.work, lambda w: evaluate_bursts(args.data, w, burst_overrides))
        print(json.dumps(result, indent=2))
        if args.write_baseline:
            _save_baseline("bursts", args.name, result)
        return 0

    root = args.labeled_set if args.labeled_set is not None else eval_dir()
    if not root.is_dir():
        raise SystemExit("the labeled set is not a folder")
    overrides = _config_overrides(args.config) if args.config is not None else {}
    overrides.update(item.split("=", 1) for item in args.set)
    if args.command == "prelabel":
        outcome = _hidden_errors(
            lambda: _with_work(
                args.work,
                lambda w: prelabel(
                    root,
                    w,
                    args.refs,
                    overrides,
                    force=args.force,
                    categories_only=args.categories_only,
                ),
            )
        )
    elif args.command == "categories":
        outcome = _hidden_errors(
            lambda: _with_work(
                args.work,
                lambda w: evaluate_categories(
                    root, w, args.refs, args.split, overrides, mistakes=args.mistakes
                ),
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
    section = args.command
    saving = getattr(args, "write_baseline", False)
    if saving and section in ("people", "categories") and section in outcome:
        _save_baseline(section, args.name, outcome)
    return 0


if __name__ == "__main__":
    sys.exit(main())
