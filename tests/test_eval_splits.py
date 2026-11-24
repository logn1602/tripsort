"""Copies stay on one side of the split. Photos tripsort keeps apart but that are twins for
labeling (shots of one burst, a near copy by pHash, the same face) and a smaller copy of a
video share the split of the smallest SHA-256 among them. File names never reach the output.
"""

import csv
import json
from pathlib import Path

import fake_faces
import pytest
from factories import camera_exif, jpeg
from fake_clip import HUES, ColorImageEncoder, WordTextEncoder
from fake_faces import MarkerFaceAnalyzer, face_photo, person_vector
from test_eval_categories import PROMPTS, run
from test_video import clip

from eval.labels import copy_groups, face_copies, split_for, split_groups


def test_groups_join_through_others() -> None:
    assert copy_groups([5, 1, 3, 2, 4], [(5, 3), (3, 1), (9, 2)]) == [[1, 3, 5], [2], [4]]
    assert copy_groups([2, 1], []) == [[1], [2]]


def test_a_group_takes_the_split_of_its_smallest_sha256() -> None:
    shas = {1: "00000001" + "f" * 56, 2: "00000002" + "0" * 56, 3: "00000003" + "0" * 56}
    assert [split_for(shas[i]) for i in (1, 2, 3)] == ["test", "dev", "test"]
    assert split_groups(shas, []) == {1: "test", 2: "dev", 3: "test"}
    # joined through 2, all three take 1's split, whatever the order of the pairs
    assert split_groups(shas, [(3, 2), (2, 1)]) == dict.fromkeys(shas, "test")
    assert split_groups(shas, [(2, 1), (2, 3)]) == dict.fromkeys(shas, "test")
    assert split_groups(shas, [(3, 2)]) == {1: "test", 2: "dev", 3: "dev"}


def test_a_shared_face_joins_copies_never_two_shots() -> None:
    shots = {
        1: "2026:07:14 10:00:00|Google|Pixel 7",
        2: "2026:07:14 10:00:05|Google|Pixel 7",  # 5 seconds later: another shot
        3: "2026:07:14 10:00:00|Google|Pixel 7",  # 1, edited: the EXIF kept
        6: "2026:07:14 10:00:00|Apple|iPhone 15",  # the same second on another phone
    }
    # 4 and 5 record no shot: copies sent on, their EXIF dropped
    likeness = {
        (1, 2): 0.97,
        (1, 3): 0.93,
        (1, 4): 0.95,
        (1, 6): 0.96,
        (2, 4): 0.96,
        (2, 5): 0.92,
        (4, 5): 0.91,
    }
    # 4 is a copy of the one photo it is most like (2), so it never chains 1 and 2 together;
    # 5's original is 2 as well (0.92 against 0.91 for 4)
    assert face_copies(likeness, shots) == {(1, 3), (2, 4), (2, 5)}
    # on a tie, the photo first in order is the original, whatever the order of the pairs
    tied = {(2, 4): 0.95, (1, 4): 0.95}
    assert face_copies(tied, shots) == face_copies(dict(reversed(tied.items())), shots)
    assert face_copies(tied, shots) == {(1, 4)}


def splits_by_stem(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as fp:
        return {Path(r["file"]).stem: r["split"] for r in csv.DictReader(fp)}


def prompts_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.toml"
    path.write_text(PROMPTS, encoding="utf-8")
    return path


def test_twins_tripsort_keeps_apart_share_a_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    marker_faces: MarkerFaceAnalyzer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flat = tmp_path / "set"
    flat.mkdir()
    # a near copy by pHash that tripsort keeps apart: same-size PNGs are never copies
    still = face_photo([], seed=60)
    still.save(flat / "kind_png_a.png")
    touched = still.copy()
    touched.paste((0, 0, 0), (0, 0, 12, 12))
    touched.save(flat / "kind_png_b.png")
    # a burst: one camera, a second apart, alike to CLIP but not by pHash
    for i in range(2):
        (flat / f"kind_burst_{i}.jpg").write_bytes(
            jpeg(face_photo([], seed=70 + i), exif=camera_exif(f"2026:07:14 10:00:0{i}"))
        )
    # one face in two photos with no EXIF, as a photo and its crop sent on share it
    monkeypatch.setitem(fake_faces.VECTORS, 8, person_vector(7))
    face_photo([(7, 0.3, 0.2, 0.4)], seed=80).save(flat / "kind_face_a.jpg")
    face_photo([(8, 0.5, 0.3, 0.3)], seed=81).save(flat / "kind_face_b.jpg")
    for person in (1, 2, 3):  # nobody's twins
        face_photo([(person, 0.3, 0.2, 0.4)], seed=90 + person).save(
            flat / f"kind_solo_{person}.jpg"
        )

    args = ["--labeled-set", flat, "--work", tmp_path / "work", "--config", prompts_file(tmp_path)]
    code, out = run(capsys, "prelabel", *args)
    assert code == 0
    report = json.loads(out)
    assert report["rows"] == 9
    assert report["split_copies"] == {
        "copy_groups": 3,
        "rows_in_copy_groups": 6,
        "largest_copy_group": 2,
        "pairs": {"burst": 1, "near_copy": 1, "same_face": 1},
    }
    split = splits_by_stem(flat / "labels.csv")
    for twins in ("png", "burst", "face"):
        first, second = (split[stem] for stem in split if stem.startswith(f"kind_{twins}_"))
        assert first == second, twins

    # run again, labels.csv is kept as it was: its split is the one that counts
    code, out = run(capsys, "prelabel", *args)
    assert (json.loads(out)["labels_csv"], json.loads(out)["split_copies"]) == (
        "kept as it was",
        None,
    )


def test_a_smaller_copy_of_a_video_shares_its_split(
    tmp_path: Path,
    color_clip: ColorImageEncoder,
    word_text: WordTextEncoder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flat = tmp_path / "set"
    flat.mkdir()
    face_photo([], seed=1).save(flat / "kind_photo.jpg")
    view = HUES["scenery"][0]
    (flat / "kind_clip.mp4").write_bytes(clip(4.0, view))
    (flat / "kind_clip_small.mp4").write_bytes(clip(4.0, view, size=(160, 120)))  # sent on
    (flat / "kind_longer.mp4").write_bytes(clip(8.0, view))  # the same view, filmed longer
    (flat / "kind_animal.mp4").write_bytes(clip(4.0, HUES["animals"][0]))  # as long, unlike

    args = ["--labeled-set", flat, "--work", tmp_path / "work", "--config", prompts_file(tmp_path)]
    code, out = run(capsys, "prelabel", *args, "--videos")
    assert code == 0
    made = json.loads(out)
    assert made["videos"] == 4
    assert made["split_copies"] == {
        "copy_groups": 1,
        "rows_in_copy_groups": 2,
        "largest_copy_group": 2,
        "pairs": {"same_video": 1},
    }
    split = splits_by_stem(flat / "video_labels.csv")
    assert split["kind_clip"] == split["kind_clip_small"]
