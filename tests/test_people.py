"""People end to end, with the marker face analyzer (see fake_faces.py) instead of the models."""

import json
from pathlib import Path
from typing import Any

import pytest
from factories import camera_exif, jpeg, make_zip, whatsapp_copy
from fake_faces import MarkerFaceAnalyzer, face_photo
from typer.testing import CliRunner

import tripsort.faces.analyzer
from tripsort.api import Project
from tripsort.cli import app
from tripsort.errors import ModelError, PeopleError, UserError

runner = CliRunner()
FRONT = camera_exif(lens="iPhone 15 front camera 2.69mm f/1.9", make="Apple", model="iPhone 15")


def shot(
    faces: list[tuple[int, float, float, float]], seed: int, exif: bytes | None = None
) -> bytes:
    return jpeg(face_photo(faces, seed=seed), exif=exif)


def solo(person: int, seed: int, exif: bytes | None = None) -> bytes:
    return shot([(person, 0.35, 0.2, 0.3)], seed, exif)


def group(people: list[int], seed: int) -> bytes:
    return shot([(p, 0.05 + 0.3 * i, 0.3, 0.25) for i, p in enumerate(people)], seed)


@pytest.fixture
def faces_trip(tmp_path: Path) -> dict[str, Path]:
    """Person 0 in 6 photos, person 1 in 5 (one a selfie), person 2 in 4, a background
    stranger (5), someone in one photo only (6), a photo without faces, and a WhatsApp copy."""
    a1 = solo(0, 1)
    mine = make_zip(
        tmp_path / "in" / "mine.zip",
        [
            ("a1.jpg", a1),
            ("a2.jpg", solo(0, 2)),
            ("a3.jpg", solo(0, 3)),
            ("a4.jpg", solo(1, 4)),
            ("a5.jpg", solo(1, 5, FRONT)),
            ("a6.jpg", group([0, 1], 6)),
            ("a7.jpg", group([0, 1, 2], 7)),
            ("a8.jpg", shot([(2, 0.3, 0.2, 0.3), (5, 0.8, 0.1, 0.02)], 8)),  # stranger behind
            ("a9.jpg", shot([], 9)),
        ],
    )
    sam = make_zip(
        tmp_path / "in" / "sam.zip",
        [
            ("b1.jpg", solo(2, 11)),
            ("b2.jpg", group([0, 2], 12)),
            ("b3.jpg", whatsapp_copy(a1)),
            ("b4.jpg", solo(1, 14)),
            ("b5.jpg", solo(6, 15)),
        ],
    )
    return {"mine": mine, "sam": sam}


def members(folder: Path) -> set[str]:
    """The original names of the files in an export folder: '..._mine_a1.jpg' -> 'a1.jpg'."""
    return {p.name.rsplit("_", 1)[-1] for p in folder.iterdir()} if folder.is_dir() else set()


def people(project: Project) -> dict[str, int]:
    return {p.name: p.photos for p in project.people.all()}


def manifest(project: Project) -> dict[str, Any]:
    return json.loads((project.folder / "manifest.json").read_text(encoding="utf-8"))


def invoke(*args: str | Path):  # type: ignore[no-untyped-def]
    return runner.invoke(app, [str(a) for a in args])


def test_people_and_shot_types(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer
) -> None:
    ingested = project.ingest([faces_trip["mine"], faces_trip["sam"]])
    assert (ingested.faces_scanned, ingested.faces_pending, ingested.face_problem) == (14, 0, None)
    stats = project.analyze().people
    assert (stats.people, stats.named, stats.unassigned) == (3, 0, 1)  # person 6: one photo
    assert stats.shots == {"solo": 9, "group": 3, "none": 1, "selfie": 1}
    assert people(project) == {"person_01": 6, "person_02": 5, "person_03": 4}

    result = project.export()
    out = project.folder
    assert members(out / "people" / "person_01") == {
        "a1.jpg",
        "a2.jpg",
        "a3.jpg",
        "a6.jpg",
        "a7.jpg",
        "b2.jpg",
    }
    assert members(out / "people" / "person_03") == {"a7.jpg", "a8.jpg", "b1.jpg", "b2.jpg"}
    assert members(out / "shots" / "selfie") == {"a5.jpg"}
    assert members(out / "shots" / "group") == {"a6.jpg", "a7.jpg", "b2.jpg"}
    assert "a8.jpg" in members(out / "shots" / "solo")  # the stranger behind does not count
    assert not (out / "shots" / "none").exists()
    for link in (out / "people" / "person_01").iterdir():
        assert link.samefile(out / "all" / link.name)  # hardlinks: no extra disk
    assert result.people_folders == 3
    assert result.shots == {"solo": 9, "group": 3, "selfie": 1}
    assert "people/ holds 3 folders" in result.summary()

    again = project.export()
    assert again.written == 0
    assert again.removed == 0


def test_a_whatsapp_copy_is_not_counted_twice(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer
) -> None:
    project.ingest([faces_trip["mine"], faces_trip["sam"]])
    project.export()
    data = manifest(project)
    copy = next(f for f in data["files"] if f["occurrences"][0]["member"] == "b3.jpg")
    assert copy["copy_of"]
    assert copy["people"] == []
    assert copy["shot"] is None
    original = next(f for f in data["files"] if f["occurrences"][0]["member"] == "a1.jpg")
    assert original["people"] == ["person_01"]
    assert original["shot"] == "solo"
    assert original["selfie"] is False
    assert original["faces"] == {"foreground": 1, "background": 0}
    stranger = next(f for f in data["files"] if f["occurrences"][0]["member"] == "a8.jpg")
    assert stranger["faces"] == {"foreground": 1, "background": 1}
    assert data["summary"]["people"] == 3
    assert data["summary"]["shots"] == {"solo": 9, "group": 3, "none": 1, "selfie": 1}
    assert data["summary"]["faces_not_scanned"] == 0
    assert [p["name"] for p in data["people"]] == ["person_01", "person_02", "person_03"]
    assert data["analyzers"]["faces"].startswith("yunet-2026may+sface-2021dec")


def test_renames_and_merges_survive_new_photos(
    faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer, tmp_path: Path
) -> None:
    looks = make_zip(  # person 3 and person 4 are one friend, with and without sunglasses
        tmp_path / "in" / "looks.zip",
        [(f"c{i}.jpg", solo(3 if i < 3 else 4, 20 + i)) for i in range(6)],
    )
    folder = tmp_path / "small"  # each look is in 3 photos, under the default of 4
    folder.mkdir()
    (folder / "tripsort.toml").write_text("[people]\nmin_cluster_size = 3\n", encoding="utf-8")
    with Project.open(folder) as project:
        check_renames_and_merges(project, faces_trip, looks, tmp_path)


def check_renames_and_merges(
    project: Project, faces_trip: dict[str, Path], looks: Path, tmp_path: Path
) -> None:
    project.ingest([faces_trip["mine"], faces_trip["sam"], looks])
    project.analyze()
    assert people(project)["person_04"] == 3
    assert people(project)["person_05"] == 3
    project.people.rename("person_01", "alex")
    merged = project.people.merge("person_05", "person_04")
    assert merged.photos == 6
    project.export()
    out = project.folder
    assert members(out / "people" / "alex") == {
        "a1.jpg",
        "a2.jpg",
        "a3.jpg",
        "a6.jpg",
        "a7.jpg",
        "b2.jpg",
    }
    assert not (out / "people" / "person_01").exists()
    assert not (out / "people" / "person_05").exists()

    more = make_zip(
        tmp_path / "in" / "more.zip",
        [
            ("d1.jpg", solo(0, 41)),
            ("d2.jpg", solo(4, 42)),
            ("d3.jpg", solo(7, 43)),
            ("d4.jpg", solo(7, 44)),
            ("d5.jpg", solo(7, 45)),
        ],
    )
    project.ingest([more])
    project.export()
    found = people(project)
    assert found["alex"] == 7  # the name stayed with the person, who gained d1
    assert found["person_04"] == 7  # still merged, and gained d2
    assert found["person_06"] == 3  # a new person gets a new number: 5 is never reused
    assert "person_05" not in found
    assert members(out / "people" / "person_06") == {"d3.jpg", "d4.jpg", "d5.jpg"}


def test_refs_name_people(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer, tmp_path: Path
) -> None:
    refs = tmp_path / "refs"
    (refs / "kim").mkdir(parents=True)
    face_photo([(1, 0.3, 0.2, 0.5)], seed=50).save(refs / "sam.jpg")
    face_photo([(9, 0.3, 0.2, 0.5)], seed=51).save(refs / "kim" / "one.jpg")  # not on the trip
    face_photo([], seed=52).save(refs / "nobody.jpg")
    (refs / "notes.txt").write_text("not a photo")
    project.ingest([faces_trip["mine"], faces_trip["sam"]])

    result = project.people.name_from_refs(refs)
    assert result.names == {"sam": "named", "kim": "no match", "nobody": "no usable photo"}
    assert "nobody.jpg: no face found" in result.problems
    assert "refs/ named 1 of 3 people." in result.summary()
    assert project.people.get("sam").photos == 5
    assert project.people.get("sam").named_by == "refs"
    assert result.analysis.people.refs_unmatched == ["kim"]

    project.people.rename("sam", "Samuel")  # the user's name wins over the ref's
    again = project.people.name_from_refs(refs)
    assert again.names["sam"] == "named by you"
    assert project.people.get("Samuel").named_by == "user"
    calls = marker_faces.calls
    project.people.name_from_refs(refs)
    assert marker_faces.calls == calls  # unchanged ref photos are not decoded again


def test_named_people_count_in_the_background(
    project: Project, marker_faces: MarkerFaceAnalyzer, tmp_path: Path
) -> None:
    trip = make_zip(
        tmp_path / "in" / "trip.zip",
        [
            ("s1.jpg", solo(1, 1)),
            ("s2.jpg", solo(1, 2)),
            ("s3.jpg", solo(1, 3)),
            ("s4.jpg", solo(1, 5)),
            ("x.jpg", shot([(0, 0.3, 0.2, 0.35), (1, 0.8, 0.1, 0.06)], 4)),  # 1 far behind 0
        ],
    )
    project.ingest([trip])
    project.analyze()
    assert project.people.get("person_01").photos == 4  # unnamed: background faces wait
    project.people.rename("person_01", "sam")
    project.export()
    assert project.people.get("sam").photos == 5
    assert members(project.folder / "people" / "sam") == {
        "s1.jpg",
        "s2.jpg",
        "s3.jpg",
        "s4.jpg",
        "x.jpg",
    }
    assert "x.jpg" in members(project.folder / "shots" / "solo")  # still one foreground face


def test_faces_are_scanned_later_when_the_models_were_missing(
    project: Project, faces_trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("could not download face_detection_yunet_2026may.onnx")

    monkeypatch.setattr(tripsort.faces.analyzer, "load_face_analyzer", offline)
    first = project.ingest([faces_trip["mine"]])
    assert first.face_problem is not None
    assert first.faces_pending == 9
    assert "Faces were not scanned" in first.summary()
    assert "9 photos still wait" in first.summary()
    assert project.analyze().people.not_scanned == 9

    analyzer = MarkerFaceAnalyzer()
    monkeypatch.setattr(
        tripsort.faces.analyzer, "load_face_analyzer", lambda progress=None: analyzer
    )
    caught_up = project.ingest()  # no zips: only the face scans
    assert (caught_up.faces_caught_up, caught_up.faces_pending) == (9, 0)
    assert "Scanned 9 photos from earlier zips for faces." in caught_up.summary()
    assert project.analyze().people.people == 2
    assert project.ingest().faces_scanned == 0  # nothing left to scan


def test_nothing_to_scan_loads_no_models(
    project: Project, faces_trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[int] = []
    analyzer = MarkerFaceAnalyzer()

    def load(progress: object = None) -> MarkerFaceAnalyzer:
        loads.append(1)
        return analyzer

    monkeypatch.setattr(tripsort.faces.analyzer, "load_face_analyzer", load)
    project.ingest([faces_trip["mine"]])
    with Project.open(project.folder) as again:
        again.ingest([faces_trip["mine"]])  # already indexed and scanned
    assert loads == [1]


def test_a_case_only_rename_respells_the_folder(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer
) -> None:
    project.ingest([faces_trip["mine"], faces_trip["sam"]])
    project.people.rename("person_02", "sam")
    project.export()
    before = sorted(p.name for p in (project.folder / "people" / "sam").iterdir())
    project.people.rename("sam", "Sam")
    result = project.export()
    assert not result.problems
    assert [p.name for p in (project.folder / "people").iterdir() if p.name.lower() == "sam"] == [
        "Sam"
    ]
    assert sorted(p.name for p in (project.folder / "people" / "Sam").iterdir()) == before


def test_threshold_changes_need_no_zips(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer
) -> None:
    project.ingest([faces_trip["mine"], faces_trip["sam"]])
    project.export()
    for zip_path in faces_trip.values():
        zip_path.unlink()
    config = "[people]\nforeground_min_ratio = 0.9\n"  # a8's stranger is too small either way
    (project.folder / "tripsort.toml").write_text(config, encoding="utf-8")
    with Project.open(project.folder) as again:
        assert again.analysis_is_stale()
        assert not again.export().problems
    config = "[people]\nmin_face_score = 0.99\n"  # the fake's faces score 0.95: nobody counts
    (project.folder / "tripsort.toml").write_text(config, encoding="utf-8")
    with Project.open(project.folder) as again:
        result = again.export()
        assert not result.problems
        assert result.removed > 0
        assert people(again) == {}
    assert not (project.folder / "people").exists()
    assert not (project.folder / "shots").exists()


def test_people_errors(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer
) -> None:
    project.ingest([faces_trip["mine"], faces_trip["sam"]])
    project.analyze()
    with pytest.raises(PeopleError, match="nobody is called"):
        project.people.get("person_99")
    with pytest.raises(PeopleError, match="merged into themselves"):
        project.people.merge("person_01", "person_01")
    project.people.rename("person_01", "alex")
    with pytest.raises(PeopleError, match="people merge"):
        project.people.rename("person_02", "Alex")
    with pytest.raises(PeopleError, match="unnamed people"):
        project.people.rename("person_02", "person_09")
    assert project.people.get("person_01").name == "alex"  # found by number after a rename


def test_people_commands(
    faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer, tmp_path: Path
) -> None:
    out = tmp_path / "sorted"
    assert invoke("run", faces_trip["mine"], faces_trip["sam"], "-o", out).exit_code == 0

    listed = invoke("people", "list", out)
    assert listed.exit_code == 0
    lines = listed.output.splitlines()
    assert lines[0].split() == ["NAME", "PHOTOS", "SOLO", "FACES", "NAMED", "BY"]
    assert lines[1].split() == ["person_01", "6", "3", "6", "-"]

    renamed = invoke("people", "rename", out, "person_02", "sam")
    assert renamed.exit_code == 0
    assert "person_02 is now sam (5 photos)." in renamed.output
    merged = invoke("people", "merge", out, "person_03", "sam")
    assert merged.exit_code == 0
    assert "sam, who now has 8 photos" in merged.output
    assert invoke("people", "rename", out, "person_42", "kim").exit_code == 1

    review = invoke("people", "review", out, "--per-person", "4")
    assert review.exit_code == 0
    assert "keep it private" in review.output
    page = (out / "review" / "people.html").read_text(encoding="utf-8")
    assert page.count("<figure") == 4 + 4 + 1  # two people, 4 samples each, one loose face
    assert page.count("data:image/jpeg;base64,") == 9
    assert "Faces in nobody&#x27;s folder" in page or "Faces in nobody's folder" in page

    cleaned = invoke("clean", out, "--yes")
    assert cleaned.exit_code == 0
    assert not (out / "review").exists()
    assert not (out / ".tripsort").exists()
    assert (out / "people").is_dir()  # exported folders stay


def test_offline_run_exits_2(
    faces_trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("could not download the face models")

    monkeypatch.setattr(tripsort.faces.analyzer, "load_face_analyzer", offline)
    result = invoke("run", faces_trip["mine"], "-o", tmp_path / "sorted", "--refs", tmp_path)
    assert result.exit_code == 2
    assert "--refs is not used" in result.output
    assert (tmp_path / "sorted" / "all").is_dir()  # dedupe and export still work


def test_people_folders_hold_copies_where_hardlinks_fail(
    project: Project,
    faces_trip: dict[str, Path],
    marker_faces: MarkerFaceAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(self: Path, target: Path) -> None:
        raise OSError(1, "Incorrect function")  # what exFAT says on Windows

    monkeypatch.setattr(Path, "hardlink_to", refuse)
    project.ingest([faces_trip["mine"], faces_trip["sam"]])
    result = project.export()
    link = next((project.folder / "people" / "person_01").iterdir())
    assert not link.samefile(project.folder / "all" / link.name)
    assert link.read_bytes() == (project.folder / "all" / link.name).read_bytes()
    assert result.link_fallbacks > 0
    assert "people/, shots/, and duplicates/ hold copies" in result.summary()


def test_catch_up_reports_a_moved_zip_until_it_is_ingested_again(
    project: Project, faces_trip: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def offline(progress: object = None) -> None:
        raise ModelError("offline")

    monkeypatch.setattr(tripsort.faces.analyzer, "load_face_analyzer", offline)
    project.ingest([faces_trip["mine"]])
    moved = tmp_path / "elsewhere" / "mine.zip"
    moved.parent.mkdir()
    faces_trip["mine"].rename(moved)

    analyzer = MarkerFaceAnalyzer()
    monkeypatch.setattr(
        tripsort.faces.analyzer, "load_face_analyzer", lambda progress=None: analyzer
    )
    stuck = project.ingest()
    assert stuck.faces_pending == 9
    assert [(p.source, p.member) for p in stuck.problems] == [("mine.zip", "9 photos")]
    assert "current location" in stuck.problems[0].message
    found = project.ingest([moved])  # recognized by content, so its photos are scanned now
    assert (found.faces_caught_up, found.faces_pending) == (9, 0)


def test_refs_problems_are_reported(
    project: Project, faces_trip: dict[str, Path], marker_faces: MarkerFaceAnalyzer, tmp_path: Path
) -> None:
    refs = tmp_path / "refs"
    (refs / "sam").mkdir(parents=True)
    (refs / "broken.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    face_photo([(0, 0.1, 0.2, 0.4), (1, 0.5, 0.2, 0.35)], seed=60).save(refs / "pair.jpg")
    face_photo([(2, 0.3, 0.2, 0.4)], seed=61).save(refs / "person_03.jpg")
    face_photo([(1, 0.3, 0.2, 0.4)], seed=62).save(refs / "Sam.jpg")
    face_photo([(1, 0.3, 0.2, 0.45)], seed=63).save(refs / "sam" / "second.jpg")
    (refs / ".hidden.jpg").write_bytes(b"skipped")
    project.ingest([faces_trip["mine"], faces_trip["sam"]])

    result = project.people.name_from_refs(refs)
    problems = "\n".join(result.problems)
    assert "broken.jpg: could not decode it" in problems
    assert "pair.jpg: more than one face; the largest was used" in problems
    assert "person_03.jpg" in problems
    assert "unnamed people" in problems
    assert ".hidden" not in problems
    assert set(result.names) == {"broken", "pair", "Sam"}  # 'sam/' joins 'Sam.jpg'
    assert result.names["Sam"] == "named"
    assert result.names["broken"] == "no usable photo"
    assert project.people.get("sam").photos == 5
    with pytest.raises(UserError, match="no such folder"):
        project.people.name_from_refs(tmp_path / "missing")
