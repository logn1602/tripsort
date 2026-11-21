"""The category evaluation: metrics, a category-only labeling pass on a flat folder, and the
mistakes page. File names must never reach the output (a real set's names are private)."""

import csv
import json
from pathlib import Path

import pytest
from fake_clip import HUES, ColorImageEncoder, WordTextEncoder
from PIL import Image, ImageDraw

from eval import labels as labels_module
from eval.category_metrics import category_report, flips
from eval.run_eval import main as run_eval_main

SECRET = "receipt_from_dr_quuxine"
PROMPTS = """
[categories.scenery]
prompts = ["scenery"]
[categories.animals]
prompts = ["animals"]
[categories.food]
prompts = ["food"]
[categories.documents]
prompts = ["documents"]
[categories.other]
prompts = ["anything"]
[people]
selfie_prompts = ["selfie"]
not_selfie_prompts = ["anything"]
"""


def test_category_report() -> None:
    pairs = [
        ("scenery", "scenery"),
        ("scenery", "other"),
        ("food", "food"),
        ("other", "other"),
        ("other", "animals"),  # nobody labeled animals: a wrong photo
    ]
    report = category_report(pairs, ["scenery", "animals", "food", "documents", "other"])
    per = report["per_category"]
    assert (per["scenery"]["precision"], per["scenery"]["recall"]) == (1.0, 0.5)
    assert per["scenery"]["f1"] == round(2 * 0.5 / 1.5, 4)
    assert per["documents"] == {
        "support": 0,
        "predicted": 0,
        "right": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    assert report["labeled_categories"] == ["scenery", "food"]
    assert report["macro_f1_labeled"] == round((per["scenery"]["f1"] + 1.0) / 2, 4)
    assert report["wrong_photos"] == {"animals": 1}
    assert report["accuracy"] == 0.6
    matrix = report["confusion"]["rows_labeled_columns_predicted"]
    assert matrix[0] == [1, 0, 0, 0, 1]  # scenery: once right, once other
    assert category_report([], ["food"])["macro_f1_labeled"] is None
    assert flips([1, 2, 3], [2, 3, 4, 5]) == {"right_to_wrong": 1, "wrong_to_right": 2}


def painted(kind: str, seed: int) -> Image.Image:
    """A photo of one kind: a checkerboard 40 levels either side of its HUES color, so the
    average color is the kind's, with a cell size of its own, so no two photos look like
    copies of each other to the perceptual hash."""
    color = HUES[kind][0]
    image = Image.new("RGB", (640, 480), tuple(max(0, c - 40) for c in color))
    draw = ImageDraw.Draw(image)
    light = tuple(min(255, c + 40) for c in color)
    cell = 24 + 13 * seed
    for top in range(0, 480, cell):
        for left in range(0, 640, cell):
            if (top // cell + left // cell) % 2 == 0:
                draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=light)
    return image


def run(capsys: pytest.CaptureFixture[str], *args: str | Path) -> tuple[int, str]:
    code = run_eval_main([str(a) for a in args])
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert "kind_" not in captured.out + captured.err
    return code, captured.out


def test_a_category_only_set_is_labeled_and_scored(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flat = tmp_path / "categories"  # a flat folder: no subfolder per phone
    flat.mkdir()
    kinds = ["scenery", "scenery", "food", "food", "documents", "animals"]
    for i, kind in enumerate(kinds):
        name = f"{SECRET}_{i}.jpg" if kind == "documents" else f"kind_{kind}_{i}.jpg"
        painted(kind, i).save(flat / name)
    prompts = tmp_path / "prompts.toml"
    prompts.write_text(PROMPTS, encoding="utf-8")
    work = tmp_path / "work"
    common = ["--labeled-set", flat, "--work", work, "--config", prompts]

    code, out = run(capsys, "prelabel", *common, "--categories-only")
    assert code == 0
    assert json.loads(out)["rows"] == 6
    page = (flat / "labels.html").read_text(encoding="utf-8")
    assert '"categoriesOnly":true' in page
    assert "labeled for categories only" in page
    with (flat / "labels.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert {r["shot"] for r in rows} == {"none"}
    assert {r["people"] for r in rows} == {""}
    guessed = {r["file"].rsplit("/", 1)[1]: r["category"] for r in rows}
    assert guessed[f"{SECRET}_4.jpg"] == "documents"  # tripsort's guess, to check
    assert guessed["kind_food_2.jpg"] == "food"

    for row in rows:  # the labeler: all right, except the animals photo is really other
        row["checked"], row["split"] = "y", "dev"
        if "animals" in row["file"]:
            row["category"] = "other"
    with (flat / "labels.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels_module.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    code, out = run(capsys, "categories", *common, "--mistakes")
    assert code == 0
    result = json.loads(out)
    report = result["categories"]
    assert result["checked_rows"] == 6
    assert report["per_category"]["food"]["right"] == 2
    assert report["per_category"]["scenery"]["recall"] == 1.0
    assert report["wrong_photos"] == {"animals": 1}
    assert report["labeled_categories"] == ["scenery", "food", "documents"]
    assert report["macro_f1_labeled"] == 1.0
    assert len(result["right_rows"]) == 5
    assert result["mistakes_page"] == {"rows": 1}
    mistakes = (flat / "category_mistakes.html").read_text(encoding="utf-8")
    assert "labeled <b>other</b>, tripsort <b>animals</b>" in mistakes
    assert result["settings"]["categories"]["food"] == ["food"]

    code, out = run(capsys, "categories", *common, "--split", "test")
    assert json.loads(out)["checked_rows"] == 0


def test_a_labeled_set_must_be_a_folder(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not a folder"):
        run_eval_main(["categories", "--labeled-set", str(tmp_path / "missing")])
