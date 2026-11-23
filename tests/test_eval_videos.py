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
    for i in range(5):  # persons 1 and 2, each in enough photos to be a person
        for person in (1, 2):
            image = face_photo([(person, 0.3, 0.2, 0.4)], seed=10 * person + i)
            (flat / f"kind_p{person}_{i}.jpg").write_bytes(
                jpeg(image, exif=camera_exif(f"2026:07:1{person} 1{i}:00:00"))
            )
    (flat / f"{SECRET}_friend.mp4").write_bytes(clip(5.0, HUES["documents"][0], face=1))
    (flat / "kind_stranger.mp4").write_bytes(clip(5.0, HUES["documents"][0], face=2))
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
    assert (made["videos"], made["videos_without_strip"]) == (3, 0)
    assert (made["videos_with_people"], made["videos_with_named_people"]) == (2, 1)
    assert made["refs"]["named"] == 1  # the refs report reads the index before it closes
    page = (flat / "video_labels.html").read_text(encoding="utf-8")
    assert "Save video_labels.csv" in page
    assert "Save labels.csv" not in page
    with (flat / "video_labels.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    friend = next(r for r in rows if SECRET in r["file"])
    assert friend["people"] == "sam"  # tripsort's guess, named by the ref, to check
    assert friend["category"] == "other"
    stranger = next(r for r in rows if "stranger" in r["file"])
    assert stranger["people"] == ""  # person 2 has no ref: never pre-filled as person_NN
    for row in rows:  # the labeler: the friend is "sam"; the view is animals, not other
        row["checked"], row["split"] = "y", "dev"
        # a person_NN left on a row by an older page is not anyone to score
        row["people"] = "sam;person_07" if SECRET in row["file"] else ""
    with (flat / "video_labels.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels_module.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    code, out = run(capsys, "videos", *common, "--split", "dev")
    assert code == 0
    report = json.loads(out)
    assert (report["checked_rows"], report["rows_not_found"]) == (3, 0)
    assert report["labeled_names_without_a_ref"] == 1  # person_07: counted, not scored
    assert report["people"]["micro"]["precision"] == 1.0  # the stranger is not scored
    assert report["people"]["micro"]["recall"] == 1.0
    assert report["categories"]["accuracy"] == 1.0
    code, out = run(capsys, "videos", *common, "--split", "test")
    assert json.loads(out)["checked_rows"] == 0
