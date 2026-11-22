"""The video evaluation: video_labels.csv and its page, then people and categories of the
videos of one split. File names never reach the output."""

import csv
import json
from pathlib import Path

import pytest
from factories import camera_exif, jpeg
from fake_clip import HUES, ColorImageEncoder, WordTextEncoder
from fake_faces import MarkerFaceAnalyzer, face_photo
from test_eval_categories import PROMPTS, SECRET, run
from test_video import clip

from eval import labels as labels_module


def test_videos_are_labeled_and_scored(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    marker_faces: MarkerFaceAnalyzer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flat = tmp_path / "set"
    flat.mkdir()
    for i in range(5):  # person 1, in enough photos to be a person
        image = face_photo([(1, 0.3, 0.2, 0.4)], seed=i)
        (flat / f"kind_p{i}.jpg").write_bytes(
            jpeg(image, exif=camera_exif(f"2026:07:14 1{i}:00:00"))
        )
    (flat / f"{SECRET}_friend.mp4").write_bytes(clip(5.0, HUES["documents"][0], face=1))
    (flat / "kind_view.mp4").write_bytes(clip(5.0, HUES["animals"][0]))
    prompts = tmp_path / "prompts.toml"
    prompts.write_text(PROMPTS, encoding="utf-8")
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "sam.jpg").write_bytes(jpeg(face_photo([(1, 0.3, 0.2, 0.4)], seed=99)))
    common = ["--labeled-set", flat, "--work", tmp_path / "work", "--config", prompts]
    common += ["--refs", refs]

    code, out = run(capsys, "prelabel", *common, "--videos")
    assert code == 0
    made = json.loads(out)
    assert (made["videos"], made["videos_without_strip"], made["videos_with_people"]) == (2, 0, 1)
    assert made["refs"]["named"] == 1  # the refs report reads the index before it closes
    page = (flat / "video_labels.html").read_text(encoding="utf-8")
    assert "Save video_labels.csv" in page
    assert "Save labels.csv" not in page
    with (flat / "video_labels.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    friend = next(r for r in rows if SECRET in r["file"])
    assert friend["people"] == "sam"  # tripsort's guess, named by the ref, to check
    assert friend["category"] == "other"
    for row in rows:  # the labeler: the friend is "sam"; the view is animals, not other
        row["checked"], row["split"] = "y", "dev"
        row["people"] = "sam" if SECRET in row["file"] else ""
    with (flat / "video_labels.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels_module.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    code, out = run(capsys, "videos", *common, "--split", "dev")
    assert code == 0
    report = json.loads(out)
    assert (report["checked_rows"], report["rows_not_found"]) == (2, 0)
    assert report["people"]["micro"]["precision"] == 1.0  # person_01 matched to sam
    assert report["people"]["micro"]["recall"] == 1.0
    assert report["categories"]["accuracy"] == 1.0
    code, out = run(capsys, "videos", *common, "--split", "test")
    assert json.loads(out)["checked_rows"] == 0
