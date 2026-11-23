"""The culling evaluation: which dev photos to label, the culling labels page, and reject
precision, recall, and burst agreement. File names never reach the output."""

import csv
import json
from pathlib import Path

import pytest
from factories import camera_exif, jpeg
from fake_clip import ColorImageEncoder, WordTextEncoder
from PIL import Image, ImageFilter
from test_eval_categories import PROMPTS, SECRET, painted, run

from eval import labels as labels_module
from eval.cull_eval import CullRow, ScoredPhoto, cull_report, pick_photos


def scored(
    key: str, tiles: float = 500.0, luma: float = 120.0, bright: float = 0.01
) -> ScoredPhoto:
    return ScoredPhoto(
        key, sharpness=tiles / 2, sharp_tiles=tiles, luma_mean=luma, bright_frac=bright
    )


def test_the_tail_of_each_score_plus_a_sample_of_the_rest() -> None:
    photos = [scored(f"p{i}", tiles=100.0 + i, luma=120.0 + (i % 7)) for i in range(200)]
    photos[150] = scored("black", tiles=900.0, luma=3.0)  # sharp tiles, but dark
    photos[160] = scored("white", tiles=800.0, bright=0.9)
    chosen = pick_photos(photos)
    tail = {p.key for p, sample, _ in chosen if sample == "tail"}
    rest = [(p, w) for p, sample, w in chosen if sample == "rest"]
    assert len(tail) == 50  # a quarter of 200
    assert {"p0", "p1", "black", "white"} <= tail  # the lowest of each score
    assert len(rest) == 50
    assert {w for _, w in rest} == {150 / 50}  # each stands for three of the 150 left
    assert chosen == pick_photos(photos)  # the same pick every time
    assert [p.key for p, _, _ in chosen] != sorted(p.key for p, _, _ in chosen)  # shuffled


def test_the_cull_report() -> None:
    rows = [
        CullRow("photo", "a", "dev", "tail", 1.0, "blurry"),
        CullRow("photo", "b", "dev", "tail", 1.0, "ok"),
        CullRow("photo", "c", "dev", "tail", 1.0, "dark"),
        CullRow("photo", "d", "dev", "rest", 3.0, "blurry"),  # stands for three photos
        CullRow("photo", "e", "dev", "rest", 3.0, "ok"),
        CullRow("photo", "f", "dev", "tail", 1.0, ""),  # not labeled
        CullRow("burst", "x | y", "dev", "", 1.0, "y"),
        CullRow("burst", "z | w", "dev", "", 1.0, "any"),
        CullRow("burst", "u | v", "dev", "", 1.0, "u"),
    ]
    reasons = {"a": "blurry", "b": "blurry", "c": "dark", "g": "dark"}
    report = cull_report(rows, reasons, {"x | y": "y", "z | w": "z", "u | v": "v"})
    assert report["photos_labeled"] == 5
    assert report["precision"] == round(2 / 3, 4)  # a and c are bad; b is fine
    assert report["recall_estimate"] == round(2 / 5, 4)  # a, c of a, c, and three like d
    assert report["flagged_not_labeled"] == 1  # g
    assert report["per_reason"]["blurry"] == {
        "flagged": 2,
        "labeled_bad": 1,
        "labeled_this_reason": 1,
    }
    assert report["burst_agreement"] == round(2 / 3, 4)  # y agrees, any agrees, u does not
    assert report["labels_weighted"]["blurry"] == 4.0


@pytest.mark.cull
def test_a_set_gets_culling_labels_and_scores(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flat = tmp_path / "set"
    flat.mkdir()

    def save(name: str, image: Image.Image, when: str) -> None:
        (flat / name).write_bytes(jpeg(image, exif=camera_exif(f"2026:07:14 {when}")))

    for i in range(8):
        save(f"kind_scenery_{i}.jpg", painted("scenery", i), f"1{i}:00:00")
    save(f"{SECRET}_blur.jpg", painted("food", 9).filter(ImageFilter.GaussianBlur(12)), "09:00:00")
    save("kind_dark.jpg", Image.new("RGB", (640, 480), (2, 2, 2)), "09:30:00")
    for i in range(3):  # a burst: one camera, a second apart
        save(f"kind_burst_{i}.jpg", painted("animals", 20 + i), f"08:00:0{i}")
    prompts = tmp_path / "prompts.toml"
    prompts.write_text(PROMPTS, encoding="utf-8")
    work = tmp_path / "work"
    common = ["--labeled-set", flat, "--work", work]

    code, _ = run(capsys, "prelabel", *common, "--config", prompts, "--categories-only")
    assert code == 0
    with (flat / "labels.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        row["checked"], row["split"] = "y", "dev"
    with (flat / "labels.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels_module.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    code, out = run(capsys, "cull-prelabel", *common)
    assert code == 0
    made = json.loads(out)
    assert (made["dev_photos"], made["photos"], made["bursts"]) == (13, 13, 1)
    assert made["tail"] >= 4  # at least a quarter; the rest is under 50, so all of it is shown
    assert made["photos_without_thumbnail"] == 0
    page = (flat / "cull.html").read_text(encoding="utf-8")
    assert "Save cull_labels.csv" in page
    assert 'e.key.toLowerCase() === "s"' in page  # Ctrl+S saves the labels, not the page
    with (flat / "cull_labels.csv").open(encoding="utf-8", newline="") as fp:
        labels = list(csv.DictReader(fp))
    assert {r["label"] for r in labels} == {""}  # nothing is pre-filled
    for r in labels:  # the labeler
        if r["kind"] == "burst":
            r["label"] = "any"
        elif SECRET in r["key"]:
            r["label"] = "blurry"
        else:
            r["label"] = "dark" if "dark" in r["key"] else "ok"
    with (flat / "cull_labels.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)

    code, out = run(capsys, "cull", *common)
    assert code == 0
    report = json.loads(out)
    assert report["dev_rejects"] == {"blurry": 1, "dark": 1}
    assert (report["precision"], report["recall_estimate"]) == (1.0, 1.0)
    assert report["burst_agreement"] == 1.0
    assert report["settings"]["reject_max_sharpness"] == 15.0


@pytest.mark.cull
def test_a_fresh_trip_gets_a_review_sample(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flat = tmp_path / "trip"  # no labels.csv: a trip nobody has labeled
    flat.mkdir()

    def save(name: str, image: Image.Image, when: str) -> None:
        (flat / name).write_bytes(jpeg(image, exif=camera_exif(f"2026:07:14 {when}")))

    for i in range(6):
        save(f"kind_scenery_{i}.jpg", painted("scenery", i), f"1{i}:00:00")
    save(f"{SECRET}_blur.jpg", painted("food", 9).filter(ImageFilter.GaussianBlur(12)), "09:00:00")
    save("kind_dark.jpg", Image.new("RGB", (640, 480), (2, 2, 2)), "09:30:00")
    for i in range(3):
        save(f"kind_burst_{i}.jpg", painted("animals", 20 + i), f"08:00:0{i}")
    common = ["--labeled-set", flat, "--work", tmp_path / "work", "--review"]

    code, out = run(capsys, "cull-prelabel", *common)
    assert code == 0
    assert SECRET not in out
    made = json.loads(out)
    assert (made["flagged"], made["tail"], made["bursts"]) == (2, 0, 1)
    assert made["photos"] == 11  # the two flagged and all nine others, fewer than 50
    with (flat / "cull_labels.csv").open(encoding="utf-8", newline="") as fp:
        labels = list(csv.DictReader(fp))
    assert {r["split"] for r in labels} == {"review"}
    assert '"split": "review"' in (flat / "cull.html").read_text(encoding="utf-8")
    for r in labels:  # the reviewer: the blur is right, the dark one is only moody
        if r["kind"] == "burst":
            r["label"] = r["key"].split(" | ")[0]
        else:
            r["label"] = "blurry" if SECRET in r["key"] else "ok"
    with (flat / "cull_labels.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)

    code, out = run(capsys, "cull", *common)
    assert code == 0
    report = json.loads(out)
    assert report["dev_rejects"] == {"blurry": 1, "dark": 1}
    assert report["precision"] == 0.5  # the dark photo was fine
    assert report["recall_estimate"] == 1.0
    assert report["bursts_labeled"] == 1
