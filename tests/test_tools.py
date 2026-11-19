"""The developer tools: synthetic copies, the dedupe evaluation, and the wheel smoke test."""

import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest
from factories import camera_exif, heic, jpeg, photo

from eval import metrics
from eval.metrics import Item
from eval.run_eval import evaluate_dedupe
from eval.run_eval import main as run_eval_main
from tools import make_synthetic_copies, smoke_test


def item(member: str, origin: str, role: str, cluster: str, variant: str = "x") -> Item:
    return Item(member=member, origin=origin, role=role, variant=variant, cluster=cluster)


def test_outcome_metrics() -> None:
    items = [
        item("o1", "A", "original", "o1", "original"),
        item("c1", "A", "copy", "o1", "whatsapp"),  # caught
        item("c2", "A", "copy", "c2", "telegram"),  # missed: still in all/
        item("o2", "B", "original", "o2", "original"),
        item("n1", "N", "negative", "o2", "burst_no_exif"),  # wrongly left out
        item("n2", "M", "negative", "n2", "document"),
    ]
    assert metrics.outcome(items) == {
        "precision": 0.5,
        "recall": 0.5,
        "left_out": 2,
        "wrongly_left_out": 1,
        "copies": 2,
        "copies_left_in_all": 1,
    }
    assert metrics.wrongly_left_out_by_family(items) == {
        "burst_no_exif": 1.0,
        "document": 0.0,
        "original": 0.0,
    }
    assert metrics.copies_left_in_all_by_variant(items) == {"telegram": 1.0, "whatsapp": 0.0}
    assert metrics.recall_by_variant(items) == {"telegram": 0.0, "whatsapp": 1.0}
    assert metrics.false_merges_by_family(items) == {"burst_no_exif": 1.0, "document": 0.0}
    assert metrics.keeper_accuracy(items) == 1.0
    pairwise = metrics.pairwise(items)
    assert (pairwise["precision"], pairwise["recall"]) == (0.5, round(1 / 3, 4))


def test_source_photo_merges_give_a_range() -> None:
    items = [
        item("o1", "A", "original", "o1", "original"),
        item("o2", "B", "original", "o1", "original"),  # filed under o1: a real copy, or not?
        item("c2", "B", "copy", "o1", "whatsapp"),
    ]
    assert metrics.source_photo_merges(items) == 1
    assert metrics.outcome(items)["precision"] == 0.0  # pessimistic: B vanished from all/
    optimistic = metrics.outcome(metrics.merge_source_photos(items))
    assert optimistic["precision"] == 1.0  # if o2 really was a copy of o1, nothing was lost


def test_metrics_on_empty_input() -> None:
    assert metrics.outcome([])["precision"] == 1.0
    assert metrics.pairwise([]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert metrics.keeper_accuracy([]) == 1.0


def test_distance_histogram() -> None:
    assert metrics.distance_histogram([0, 0, 2, 13, None, 40], cap=12) == {
        "0": 2,
        "2": 1,
        ">12": 2,
        "none": 1,
    }


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    argv = [str(out), "--generate", "8", "--size", "800x600", "--documents", "3"]
    assert make_synthetic_copies.main([*argv, "--negatives-every", "2"]) == 0
    return out


def test_generator_output(synthetic: Path) -> None:
    with (synthetic / "truth.csv").open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    roles = {(r["role"], r["variant"]) for r in rows}
    for variant in make_synthetic_copies.COPY_VARIANTS:
        assert ("copy", variant) in roles
    for family in make_synthetic_copies.NEGATIVE_FAMILIES:
        assert ("negative", family) in roles
    with zipfile.ZipFile(synthetic / "originals.zip") as zf:
        names = zf.namelist()
    assert all(n.startswith(("originals/o", "negatives/n", "negatives/d")) for n in names)


def test_dedupe_evaluation(synthetic: Path, tmp_path: Path) -> None:
    result = evaluate_dedupe(synthetic, tmp_path / "work")
    assert result["dedupe"]["recall"] == 1.0
    assert result["copies_left_in_all_by_variant"] == dict.fromkeys(
        sorted(make_synthetic_copies.COPY_VARIANTS), 0.0
    )
    lost = result["wrongly_left_out_by_family"]
    assert lost["original"] == 0.0
    assert lost["burst_exif"] == 0.0  # different capture times are never copies
    assert lost["burst_same_second_exposure_differs"] == 0.0  # different exposure or ISO
    assert lost["document"] == 0.0
    assert result["keeper_accuracy"] == 1.0
    text = json.dumps(result)
    assert "o0000" not in text  # numbers only: no member names


def test_eval_writes_the_baseline(
    synthetic: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import eval.run_eval as run_eval

    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(run_eval, "BASELINE", baseline)
    assert run_eval_main(["dedupe", str(synthetic), "--name", "test", "--write-baseline"]) == 0
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    assert saved["dedupe"]["test"]["dedupe"]["recall"] == 1.0
    assert json.loads(capsys.readouterr().out)["items"] > 0


def test_generator_from_a_photo_folder(tmp_path: Path) -> None:
    photos = tmp_path / "private" / "trip"
    (photos / "day1").mkdir(parents=True)
    (photos / "day1" / "my_secret_name.jpg").write_bytes(jpeg(photo(1), exif=camera_exif()))
    (photos / "IMG_2.HEIC").write_bytes(heic(photo(2, (480, 360)), exif=camera_exif()))
    (photos / "same_bytes_again.jpg").write_bytes(
        (photos / "day1" / "my_secret_name.jpg").read_bytes()
    )
    (photos / "broken.jpg").write_bytes(b"not an image")
    (photos / "notes.txt").write_text("skip me")
    out = tmp_path / "out"
    assert make_synthetic_copies.main([str(out), "--from", str(photos)]) == 0
    with zipfile.ZipFile(out / "originals.zip") as zf:
        names = zf.namelist()
    assert "originals/o00000.jpg" in names or "originals/o00000.heic" in names
    assert not any("secret" in n or "IMG_2" in n for n in names)
    text = (out / "truth.csv").read_text(encoding="utf-8")
    assert "secret" not in text
    originals = sum(1 for line in text.splitlines() if ",original," in line)
    assert originals == 2  # the exact duplicate and broken.jpg are skipped


def test_generator_with_nothing_usable(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert (
        make_synthetic_copies.main([str(tmp_path / "out"), "--from", str(tmp_path / "empty")]) == 1
    )


def test_smoke_test_passes_against_this_install(tmp_path: Path) -> None:
    smoke_test.run_smoke([sys.executable, "-m", "tripsort"], tmp_path)


def test_smoke_test_reports_failures(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke_test.main([sys.executable, "-c", "import sys; sys.exit(3)"]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_generator_holds_one_photo_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = make_synthetic_copies._generated
    still_open: set[int] = set()

    def tracking(n, size, rng):  # type: ignore[no-untyped-def]
        for original in real(n, size, rng):
            assert not still_open, "the previous photo is still decoded"
            key = id(original)
            still_open.add(key)
            close = original.image.close

            def close_and_forget(close=close, key=key):  # type: ignore[no-untyped-def]
                still_open.discard(key)
                close()

            original.image.close = close_and_forget
            yield original

    monkeypatch.setattr(make_synthetic_copies, "_generated", tracking)
    argv = [str(tmp_path / "out"), "--generate", "4", "--size", "320x240", "--documents", "1"]
    assert make_synthetic_copies.main(argv) == 0
    assert not still_open
