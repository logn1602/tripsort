"""The people evaluation on a synthetic labeled set (marker faces, made-up names).

Besides the numbers, these tests check that no file, folder, or person name ever reaches
the output: the real labeled set holds friends' photos and names.
"""

import csv
import json
from pathlib import Path

import pytest
from fake_faces import MarkerFaceAnalyzer, face_photo

from eval import labels as labels_module
from eval import people_metrics
from eval.people_metrics import Photo
from eval.run_eval import _hidden_errors
from eval.run_eval import main as run_eval_main

SECRET_FOLDERS = ("Phone of Zorblax", "Camera of Quuxine")
SECRET_PEOPLE = ("zorblax", "quuxine", "Bartholomew")


@pytest.fixture
def labeled_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "private"
    a, b = (root / name for name in SECRET_FOLDERS)
    (a / "sub").mkdir(parents=True)
    b.mkdir()
    for i in range(4):
        face_photo([(0, 0.3, 0.2, 0.3)], seed=i).save(a / f"solo0_{i}.jpg")
        face_photo([(1, 0.3, 0.2, 0.3)], seed=10 + i).save(b / f"solo1_{i}.jpg")
    face_photo([(0, 0.05, 0.3, 0.25), (1, 0.5, 0.3, 0.25)], seed=20).save(a / "sub" / "pair.jpg")
    face_photo([], seed=21).save(b / "view.jpg")
    (b / "notes.txt").write_text("ignored")
    monkeypatch.setenv("TRIPSORT_EVAL_DIR", str(root))
    return root


def run(capsys: pytest.CaptureFixture[str], *args: str | Path) -> tuple[int, str]:
    code = run_eval_main([str(a) for a in args])
    captured = capsys.readouterr()
    output = captured.out + captured.err
    for secret in (*SECRET_FOLDERS, *SECRET_PEOPLE, "solo0", "solo1", "pair.jpg", "person_0"):
        assert secret not in output, secret
    return code, captured.out


def label_everything(root: Path, names: dict[str, str]) -> None:
    """Act as the labeler: check every row, with true people by file name prefix."""
    path = root / "labels.csv"
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    for row in rows:
        stem = Path(row["file"]).stem
        if stem.startswith("solo"):
            row["people"] = names[stem[4]]
            row["shot"] = "solo"
        elif stem == "pair":
            row["people"] = f"{names['0']};{names['1']}"
            row["shot"] = "group"
        else:
            row["people"], row["shot"], row["category"] = "", "none", "scenery"
        row["checked"] = "y"
        row["category"] = row["category"] or "other"
    with path.open("w", encoding="utf-8-sig", newline="") as fp:  # as a spreadsheet saves it
        writer = csv.DictWriter(fp, fieldnames=list(labels_module.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_prelabel_then_score(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    work = tmp_path / "work"
    code, out = run(capsys, "prelabel", "--work", work)
    assert code == 0
    report = json.loads(out)
    assert report["rows"] == 10
    assert report["dev"] + report["test"] == 10
    assert (report["people_found"], report["labels_csv"], report["rows_without_photo"]) == (
        2,
        "written",
        0,
    )
    page = (labeled_set / "labels.html").read_text(encoding="utf-8")
    assert page.count("class='card'") == 10
    assert page.count("data:image/jpeg;base64,") == 10
    with (labeled_set / "labels.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert [int(r["row"]) for r in rows] == list(range(1, 11))
    assert all(r["file"].startswith(SECRET_FOLDERS) for r in rows)
    assert {r["category"] for r in rows if r["shot"] in ("solo", "group")} == {"other"}

    label_everything(labeled_set, {"0": "Zorblax", "1": "quuxine"})
    code, out = run(capsys, "prelabel", "--work", work)  # labels.csv is never overwritten
    assert json.loads(out)["labels_csv"] == "kept as it was"
    assert json.loads(out)["checked"] == 10

    for split in ("dev", "test"):
        code, out = run(capsys, "people", "--work", work, "--split", split)
        assert code == 0
        result = json.loads(out)
        assert result["checked_rows"] == sum(r["split"] == split for r in rows)
        assert result["rows_not_found"] == 0
        assert result["people"]["micro"]["precision"] in (1.0, None)
        assert result["people"]["matched_by_name"] == 0  # no refs: matched by shared photos
        assert all(p["person"].startswith("p") for p in result["people"]["per_person"])
        assert result["shots"]["accuracy"] == 1.0


def test_refs_match_labels_by_name(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    face_photo([(0, 0.3, 0.2, 0.5)], seed=40).save(refs / "Zorblax.jpg")
    face_photo([(1, 0.3, 0.2, 0.5)], seed=41).save(refs / "quuxine.jpg")
    work = tmp_path / "work"
    assert run(capsys, "prelabel", "--work", work, "--refs", refs)[0] == 0
    label_everything(labeled_set, {"0": "zorblax", "1": "Quuxine"})  # case does not matter
    baseline = tmp_path / "baseline.json"
    import eval.run_eval as run_eval

    run_eval.BASELINE = baseline
    code, out = run(
        capsys, "people", "--work", work, "--refs", refs, "--write-baseline", "--name", "t"
    )
    assert code == 0
    result = json.loads(out)
    assert result["people"]["matched_by_name"] == result["people"]["labeled_people"] > 0
    assert result["people"]["micro"] == {"precision": 1.0, "recall": 1.0}
    assert json.loads(baseline.read_text(encoding="utf-8"))["people"]["t"] == result


def test_settings_can_be_tried_without_editing_files(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    work = tmp_path / "work"
    run(capsys, "prelabel", "--work", work)
    label_everything(labeled_set, {"0": "a", "1": "b"})
    code, out = run(capsys, "people", "--work", work, "--set", "people.min_face_score=0.99")
    assert code == 0
    result = json.loads(out)
    assert result["settings"]["min_face_score"] == 0.99
    assert result["people"]["predicted_people"] == 0  # the fake's faces score 0.95


def test_label_problems_name_rows_and_columns_only(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run(capsys, "prelabel", "--work", tmp_path / "work")
    path = labeled_set / "labels.csv"
    text = path.read_text(encoding="utf-8").splitlines()
    first = text[1].split(",")
    first[5] = "Bartholomew"  # a name typed into the shot column
    text[1] = ",".join(first)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    code, out = run(capsys, "people", "--work", tmp_path / "work")
    assert code == 0
    result = json.loads(out)
    assert result["label_problems"] == 1
    assert result["first_problems"] == ["row 1: unexpected value in shot"]


def test_errors_never_print_their_message(capsys: pytest.CaptureFixture[str]) -> None:
    def fail() -> dict[str, object]:
        raise ValueError("Bartholomew's photo IMG_0001.jpg")

    assert _hidden_errors(fail) is None
    err = capsys.readouterr().err
    assert "ValueError" in err
    assert "message hidden" in err
    assert "Bartholomew" not in err
    assert "IMG_0001" not in err


def test_missing_eval_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import eval.datasets as datasets

    monkeypatch.setenv("TRIPSORT_EVAL_DIR", str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="does not name a folder"):
        datasets.eval_dir()


# metrics


def photo(labeled: set[str], predicted: set[str], shot: str = "solo", guess: str = "solo") -> Photo:
    return Photo(labeled, predicted, shot, guess, False, False)


def test_people_are_matched_by_name_then_by_shared_photos() -> None:
    photos = [
        photo({"alex", "sam"}, {"alex", "person_03"}, "group", "group"),
        photo({"sam"}, {"person_03"}),
        photo({"sam"}, {"person_04"}),  # a second, smaller folder of sam: counts as missed
        photo({"kim"}, set(), "solo", "none"),
    ]
    matching = people_metrics.match_people(photos)
    assert matching.pairs == {"alex": "alex", "person_03": "sam"}
    report = people_metrics.people_report(photos)
    per = {p["person"]: p for p in report["per_person"]}
    assert per["p1"]["photos"] == 3  # sam has the most labeled photos
    assert (per["p1"]["precision"], per["p1"]["recall"]) == (1.0, round(2 / 3, 4))
    assert per["p3"]["recall"] == 0.0
    assert per["p3"]["precision"] is None
    assert report["photos_with_unmatched_predicted_people"] == 1
    assert report["micro"] == {"precision": 1.0, "recall": 0.6}


def test_wrong_photos_lower_precision() -> None:
    photos = [photo({"sam"}, {"sam"}), photo({"alex"}, {"sam"}), photo({"sam"}, {"sam"})]
    per = people_metrics.people_report(photos)["per_person"]
    sam = next(p for p in per if p["photos"] == 2)
    assert (sam["precision"], sam["recall"]) == (round(2 / 3, 4), 1.0)


def test_shot_and_face_reports() -> None:
    photos = [
        Photo({"a"}, {"a"}, "solo", "solo", True, True, ("a", "a")),
        Photo({"b"}, {"b"}, "solo", "solo", False, True, ("b", None)),
        Photo(set(), set(), "none", None, False, False),
        Photo({"a", "b"}, {"a"}, "group", "solo", False, False),
    ]
    shots = people_metrics.shot_report(photos)
    assert shots["accuracy"] == 0.75
    assert shots["not_scanned"] == 1
    assert shots["confusion"]["rows_true_columns_predicted"] == [[2, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert shots["selfie"] == {"labeled": 1, "precision": 0.5, "recall": 1.0}
    faces = people_metrics.face_report(photos)
    assert (faces["solo_faces"], faces["unassigned"]) == (2, 0.5)
    assert faces["homogeneity"] == 1.0
