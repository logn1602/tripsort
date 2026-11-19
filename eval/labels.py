"""labels.csv: one row per kept photo of the labeled set, pre-filled with tripsort's guesses.

Columns:

* ``row``: the row number labels.html shows next to the photo;
* ``file``: ``<subfolder>/<path>`` inside the labeled set's folder;
* ``split``: ``dev`` (tune on it) or ``test`` (report on it once), 50/50 by the photo's
  SHA-256, so the split never depends on the order of anything;
* ``checked``: ``y`` once the row was reviewed; only checked rows count;
* ``people``: the labeled people in the photo, ``;``-separated: only the people with a ref
  (the labeler and the friends they chose), in every row they are in, background included.
  Everyone else is left out, and tripsort's other people go unscored;
* ``shot``: ``solo``, ``group``, or ``none``, counting everyone in the photo;
* ``selfie``: ``y`` or ``n``;
* ``category``: ``scenery``, ``animals``, ``food``, ``documents``, or ``other``.

Validation messages name the row and column only, never a value: a value may be a name.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("row", "file", "split", "checked", "people", "shot", "selfie", "category")
SHOTS = ("solo", "group", "none")
CATEGORIES = ("scenery", "animals", "food", "documents", "other")
YES = frozenset({"y", "yes", "x", "1", "true"})
NO = frozenset({"", "n", "no", "0", "false"})


@dataclass
class Row:
    row: int
    file: str
    split: str
    checked: bool
    people: list[str]
    shot: str
    selfie: bool
    category: str


def split_for(sha256: str) -> str:
    return "dev" if int(sha256[:8], 16) % 2 == 0 else "test"


def person_key(name: str) -> str:
    return " ".join(name.split()).casefold()


def write_rows(path: Path, rows: list[Row]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow(
                [
                    r.row,
                    r.file,
                    r.split,
                    "y" if r.checked else "",
                    ";".join(r.people),
                    r.shot,
                    "y" if r.selfie else "n",
                    r.category,
                ]
            )
    tmp.replace(path)


def read_rows(path: Path) -> tuple[list[Row], list[str]]:
    """The rows and the problems found (row number and column only)."""
    rows: list[Row] = []
    problems: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as fp:  # spreadsheets add a BOM
        reader = csv.DictReader(fp)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            return [], [f"the header must be: {','.join(COLUMNS)}"]
        for line, raw in enumerate(reader, start=2):
            where = f"line {line}"
            try:
                number = int(raw["row"])
            except (TypeError, ValueError):
                problems.append(f"{where}: row is not a number")
                continue
            where = f"row {number}"
            checked = (raw["checked"] or "").strip().casefold()
            selfie = (raw["selfie"] or "").strip().casefold()
            shot = (raw["shot"] or "").strip().casefold()
            category = (raw["category"] or "").strip().casefold()
            split = (raw["split"] or "").strip().casefold()
            bad = [
                column
                for column, ok in (
                    ("checked", checked in YES or checked in NO),
                    ("selfie", selfie in YES or selfie in NO),
                    ("shot", shot in SHOTS or (shot == "" and checked not in YES)),
                    ("category", category in CATEGORIES or (category == "" and checked not in YES)),
                    ("split", split in ("dev", "test")),
                )
                if not ok
            ]
            if bad:
                problems.append(f"{where}: unexpected value in {', '.join(bad)}")
                continue
            people = [p.strip() for p in (raw["people"] or "").split(";") if p.strip()]
            rows.append(
                Row(
                    number,
                    raw["file"],
                    split,
                    checked in YES,
                    people,
                    shot,
                    selfie in YES,
                    category,
                )
            )
    return rows, problems
