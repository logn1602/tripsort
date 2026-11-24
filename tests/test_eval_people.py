"""The people evaluation on a synthetic labeled set (marker faces, made-up names).

Besides the numbers, these tests check that no file, folder, or person name ever reaches
the output: the real labeled set holds friends' photos and names.
"""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import fake_faces
import pytest
from fake_faces import MarkerFaceAnalyzer, face_photo
from PIL import Image

from eval import labels as labels_module
from eval import people_metrics, run_eval
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
    different_photos_of_a_person(monkeypatch)
    return root


def different_photos_of_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake faces of one person are about 0.95 alike in any two photos, where real ones
    stayed at 0.81 or below in different photos: raise the split's same-face bar above the
    fakes, so photos of one fake person are split as different photos, not as copies."""
    monkeypatch.setattr(run_eval, "SPLIT_SAME_FACE", 0.99)


def run(capsys: pytest.CaptureFixture[str], *args: str | Path) -> tuple[int, str]:
    code = run_eval_main([str(a) for a in args])
    captured = capsys.readouterr()
    output = (captured.out + captured.err).casefold()
    for secret in (*SECRET_FOLDERS, *SECRET_PEOPLE, "solo0", "solo1", "pair.jpg", "person_0"):
        assert secret.casefold() not in output, secret
    return code, captured.out


def label_everything(root: Path, names: dict[str, str]) -> None:
    """Act as the labeler: check every row, with true people by file name prefix."""
    path = root / "labels.csv"
    with path.open(encoding="utf-8-sig", newline="") as fp:  # it may have been saved below
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
    assert (report["refs"], report["ref_copies"], report["named_people"]) == (None, None, 0)
    page = (labeled_set / "labels.html").read_text(encoding="utf-8")
    assert page.count("class='card'") == 10
    assert page.count("data:image/jpeg;base64,") == 10
    with (labeled_set / "labels.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert [int(r["row"]) for r in rows] == list(range(1, 11))
    assert all(r["file"].startswith(SECRET_FOLDERS) for r in rows)
    assert {r["category"] for r in rows if r["shot"] in ("solo", "group")} == {"other"}
    assert {r["people"] for r in rows} == {""}  # person_NN is never guessed: only names

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
    code, out = run(capsys, "prelabel", "--work", work, "--refs", refs)
    assert code == 0
    report = json.loads(out)
    assert report["refs"] == {
        "names": 2,
        "named": 2,
        "no_match": 0,
        "no_usable_photo": 0,
        "files_skipped": 0,
    }
    copies = report["ref_copies"]
    assert (copies["ref_photos"], copies["same_file"], copies["near_copy"]) == (2, 0, 0)
    assert report["named_people"] == 2
    with (labeled_set / "labels.csv").open(encoding="utf-8", newline="") as fp:
        guesses = {Path(r["file"]).stem: r["people"] for r in csv.DictReader(fp)}
    assert (guesses["solo0_0"], guesses["solo1_0"]) == ("Zorblax", "quuxine")
    assert (guesses["pair"], guesses["view"]) == ("Zorblax;quuxine", "")
    label_everything(labeled_set, {"0": "zorblax", "1": "Quuxine"})  # case does not matter
    baseline = tmp_path / "baseline.json"
    run_eval.BASELINE = baseline
    code, out = run(
        capsys, "people", "--work", work, "--refs", refs, "--write-baseline", "--name", "t"
    )
    assert code == 0
    result = json.loads(out)
    assert result["people"]["matched_by_name"] == result["people"]["labeled_people"] > 0
    assert result["people"]["micro"] == {"precision": 1.0, "recall": 1.0}
    assert result["labeled_names_without_a_ref"] == 0
    assert result["ref_copies"]["same_file"] == 0
    assert json.loads(baseline.read_text(encoding="utf-8"))["people"]["t"] == result


def test_refs_that_copy_a_labeled_photo_are_flagged(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refs = tmp_path / "refs"
    (refs / "Zorblax").mkdir(parents=True)
    photos = labeled_set / SECRET_FOLDERS[0]
    (refs / "Zorblax" / "1.jpg").write_bytes((photos / "solo0_0.jpg").read_bytes())
    with Image.open(photos / "solo0_1.jpg") as image:  # a smaller copy, saved again
        image.resize((960, 720)).save(refs / "Zorblax" / "2.jpg", quality=80)
    face_photo([(5, 0.3, 0.2, 0.5)], seed=42).save(refs / "quuxine.jpg")  # nobody in the set
    code, out = run(capsys, "prelabel", "--work", tmp_path / "work", "--refs", refs)
    assert code == 0
    copies = json.loads(out)["ref_copies"]
    assert (copies["ref_photos"], copies["same_file"], copies["near_copy"]) == (3, 1, 2)
    assert (copies["closest_phash_distance"], copies["highest_face_similarity"]) == (0, 1.0)
    # by position only: quuxine is the 1st name (ignoring case), Zorblax the 2nd
    assert copies["flagged"] == [
        {
            "name_position": 2,
            "photo_position": 1,
            "photos_for_name": 2,
            "why": ["same file", "near copy", "same face"],
        },
        {
            "name_position": 2,
            "photo_position": 2,
            "photos_for_name": 2,
            "why": ["near copy", "same face"],
        },
    ]


def test_every_ref_name_is_a_labeled_person(
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
    run(capsys, "prelabel", "--work", work, "--refs", refs)
    label_everything(labeled_set, {"0": "Zorblax", "1": ""})  # no checked row lists quuxine
    results = []
    for split in ("dev", "test"):
        code, out = run(capsys, "people", "--work", work, "--refs", refs, "--split", split)
        assert code == 0
        results.append(json.loads(out))
    for result in results:
        assert result["people"]["labeled_people"] == 2
        assert result["people"]["other_predicted_people"] == 0
        assert result["labeled_names_without_a_ref"] == 0
    # tripsort names person 1 quuxine: each checked photo it puts quuxine on is a wrong photo
    # (quuxine is p2 in both splits: no labeled photos, against Zorblax's 5)
    quuxine = [p for r in results for p in r["people"]["per_person"] if p["person"] == "p2"]
    assert [p["photos"] for p in quuxine] == [0, 0]
    assert {p["precision"] for p in quuxine} - {None} == {0.0}
    assert {p["recall"] for p in quuxine} == {None}

    label_everything(labeled_set, {"0": "Zorblx", "1": ""})  # a typo: a name with no ref
    code, out = run(capsys, "people", "--work", work, "--refs", refs)
    assert code == 0
    result = json.loads(out)
    assert result["labeled_names_without_a_ref"] == 1
    assert result["people"]["labeled_people"] == 3


def test_people_without_a_ref_are_left_unscored(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    face_photo([(0, 0.3, 0.2, 0.5)], seed=40).save(refs / "Zorblax.jpg")
    work = tmp_path / "work"
    code, out = run(capsys, "prelabel", "--work", work, "--refs", refs)
    assert code == 0
    assert json.loads(out)["named_people"] == 1
    with (labeled_set / "labels.csv").open(encoding="utf-8", newline="") as fp:
        assert {r["people"] for r in csv.DictReader(fp)} == {"Zorblax", ""}
    label_everything(labeled_set, {"0": "Zorblax", "1": ""})  # only the person with a ref
    results = []
    for split in ("dev", "test"):
        code, out = run(capsys, "people", "--work", work, "--refs", refs, "--split", split)
        assert code == 0
        results.append(json.loads(out))
    for result in results:
        assert result["people"]["labeled_people"] == 1
        assert result["people"]["micro"]["precision"] in (1.0, None)
        assert result["people"]["micro"]["recall"] in (1.0, None)
    assert sum(r["people"]["other_predicted_people"] for r in results) >= 1
    assert sum(r["faces"]["solo_faces"] for r in results) == 4  # the solo photos of Zorblax


def test_suggestions_are_scored_on_dev_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_faces: MarkerFaceAnalyzer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zorblax in two looks (0 and 3), quuxine (1), and a stranger like quuxine (5)."""
    e = fake_faces.orthonormal([0, 1, 3, 5])
    monkeypatch.setitem(fake_faces.VECTORS, 1, e[1])
    monkeypatch.setitem(fake_faces.VECTORS, 3, 0.72 * e[0] + 0.694 * e[3])
    monkeypatch.setitem(fake_faces.VECTORS, 5, 0.72 * e[1] + 0.694 * e[5])
    root = tmp_path / "private"
    for folder, people in zip(SECRET_FOLDERS, ((0, 3), (1, 5)), strict=True):
        (root / folder).mkdir(parents=True)
        for person in people:
            for i in range(4):
                face_photo([(person, 0.3, 0.2, 0.3)], seed=10 * person + i).save(
                    root / folder / f"solo{person}_{i}.jpg"
                )
    monkeypatch.setenv("TRIPSORT_EVAL_DIR", str(root))
    refs = tmp_path / "refs"
    refs.mkdir()
    face_photo([(0, 0.3, 0.2, 0.5)], seed=40).save(refs / "Zorblax.jpg")
    face_photo([(1, 0.3, 0.2, 0.5)], seed=41).save(refs / "quuxine.jpg")
    work = tmp_path / "work"
    run(capsys, "prelabel", "--work", work, "--refs", refs)
    label_everything(root, {"0": "Zorblax", "3": "Zorblax", "1": "quuxine", "5": ""})
    path = root / "labels.csv"
    with path.open(encoding="utf-8-sig", newline="") as fp:
        rows = [{**r, "split": "dev"} for r in csv.DictReader(fp)]  # every row in dev
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(labels_module.COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    code, out = run(capsys, "people", "--work", work, "--refs", refs)
    assert code == 0
    result = json.loads(out)
    assert result["people"]["micro"] == {"precision": 1.0, "recall": 0.6667}  # one look missed
    found = result["suggestions"]
    assert (found["shown"], found["with_two_candidates"], found["confirmed"]) == (2, 0, 1)
    scored = sorted(
        (c["named"], c["right"], c["wrong"], c["confirmed"]) for c in found["candidates"]
    )
    assert scored == [("p1", 4, 0, True), ("p2", 0, 4, False)]  # the stranger is not quuxine
    assert all(0.65 < c["similarity"] < 0.8 for c in found["candidates"])
    confirmed, everything = found["after_accepting_confirmed"], found["after_accepting_all"]
    assert (confirmed["merged"], confirmed["micro"]) == (1, {"precision": 1.0, "recall": 1.0})
    assert (everything["merged"], everything["micro"]) == (2, {"precision": 0.75, "recall": 1.0})
    by_person = {p["person"]: p["precision"] for p in everything["per_person"]}
    assert by_person == {"p1": 1.0, "p2": 0.5}  # what accepting blindly costs

    code, out = run(capsys, "people", "--work", work, "--refs", refs, "--split", "test")
    assert code == 0
    empty = json.loads(out)  # no test rows, yet what to accept still comes from dev
    assert empty["checked_rows"] == 0
    assert empty["suggestions"]["confirmed"] == 1


def page_data(root: Path) -> dict[str, Any]:
    """The data labels.html's script starts from (the page shows made-up photos here)."""
    text = (root / "labels.html").read_text(encoding="utf-8")
    found = re.search(
        r"<script type='application/json' id='labels-data'>(.*?)</script>", text, re.S
    )
    assert found is not None
    data: dict[str, Any] = json.loads(found.group(1))
    return data


def test_the_page_starts_from_labels_csv(
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
    code, out = run(capsys, "prelabel", "--work", work, "--refs", refs)
    assert (code, json.loads(out)["name_buttons"]) == (0, 2)
    first = page_data(labeled_set)
    assert (first["names"], first["canSave"]) == (["quuxine", "Zorblax"], True)
    label_everything(labeled_set, {"0": "Zorblax", "1": "quuxine"})
    run(capsys, "prelabel", "--work", work, "--refs", refs)  # writes only the page again
    rows, _ = labels_module.read_rows(labeled_set / "labels.csv")
    data = page_data(labeled_set)
    assert [
        (r["row"], r["file"], r["checked"], r["people"], r["shot"], r["selfie"], r["category"])
        for r in data["rows"]
    ] == [(r.row, r.file, r.checked, r.people, r.shot, r.selfie, r.category) for r in rows]
    assert data["set"] == first["set"]  # the same rows
    assert data["base"] != first["base"]  # with the labels from the file
    csv_bytes = (labeled_set / "labels.csv").read_bytes()
    assert data["csvSha256"] == hashlib.sha256(csv_bytes).hexdigest()


def test_the_page_never_saves_over_rows_it_could_not_read(
    labeled_set: Path,
    marker_faces: MarkerFaceAnalyzer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run(capsys, "prelabel", "--work", tmp_path / "work")
    path = labeled_set / "labels.csv"
    text = path.read_text(encoding="utf-8").splitlines()
    first = text[1].split(",")
    first[5] = "sideways"  # not a shot
    text[1] = ",".join(first)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    code, out = run(capsys, "prelabel", "--work", tmp_path / "work")
    assert (code, json.loads(out)["label_problems"]) == (0, 1)
    data = page_data(labeled_set)
    assert (data["canSave"], len(data["rows"])) == (False, 9)


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
    assert report["other_predicted_people"] == 1  # person_04: counted, not scored
    assert report["micro"] == {"precision": 1.0, "recall": 0.6}


def test_people_nobody_labeled_are_not_scored() -> None:
    photos = [
        photo({"alex"}, {"alex", "person_05"}, "group", "group"),
        photo(set(), {"person_05"}),  # a solo photo of someone without a ref
        photo({"alex"}, {"alex"}),
    ]
    report = people_metrics.people_report(photos)
    assert (report["labeled_people"], report["other_predicted_people"]) == (1, 1)
    assert report["micro"] == {"precision": 1.0, "recall": 1.0}


def test_labeled_people_are_labeled_in_every_row() -> None:
    # kim's labeled photos are all in the other split; a photo here that does not list kim
    # is still a wrong photo in kim's folder
    photos = [photo({"alex"}, {"alex", "kim"}, "group", "group"), photo({"alex"}, {"alex"})]
    report = people_metrics.people_report(photos, Counter({"alex": 5, "kim": 2}))
    per = {p["person"]: p for p in report["per_person"]}
    assert (per["p1"]["photos"], per["p1"]["precision"], per["p1"]["recall"]) == (2, 1.0, 1.0)
    assert (per["p2"]["photos"], per["p2"]["precision"], per["p2"]["recall"]) == (0, 0.0, None)
    assert report["other_predicted_people"] == 0


def test_solo_faces_come_from_solo_photos_of_labeled_people() -> None:
    solo_face = people_metrics.solo_face
    assert solo_face({"alex"}, "solo", ["alex"]) == ("alex", "alex")
    assert solo_face({"alex"}, "solo", [None]) == ("alex", None)
    assert solo_face(set(), "solo", ["person_05"]) is None  # someone nobody labeled
    # alex and someone unlabeled with one face missed: the face found may not be alex's
    assert solo_face({"alex"}, "group", ["person_05"]) is None
    assert solo_face({"alex"}, "solo", ["alex", "person_05"]) is None


def test_aliases_follow_photo_counts_past_nine_people() -> None:
    labeled = [f"friend{i:02d}" for i in range(12)]
    photos = [photo({name}, {name}) for i, name in enumerate(labeled) for _ in range(12 - i)]
    per = people_metrics.people_report(photos)["per_person"]
    assert [p["person"] for p in per] == [f"p{i}" for i in range(1, 13)]
    assert [p["photos"] for p in per] == list(range(12, 0, -1))


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
