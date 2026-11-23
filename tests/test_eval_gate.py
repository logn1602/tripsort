"""The CI eval gate (`eval/run_eval.py gate`): public metrics against eval/baseline.json."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import eval.run_eval as run_eval
from eval.run_eval import GATE_TOLERANCE, GATED, compare_to_baseline, gate_report

BASELINE = json.loads(run_eval.BASELINE.read_text(encoding="utf-8"))


def gated_results(baseline: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(s, k): copy.deepcopy(baseline[s][k]) for s, k in GATED}


def _set(result: dict[str, Any], dotted: str, value: object) -> None:
    *path, last = dotted.split(".")
    for key in path:
        result = result[key]
    result[last] = value


def test_the_committed_baseline_has_every_gated_number() -> None:
    for (section, key), (size, metrics) in GATED.items():
        entry = BASELINE[section][key]
        assert isinstance(entry[size], int), (section, key)
        for metric in metrics:
            value = run_eval._lookup(entry, metric)
            assert isinstance(value, float | int), (section, key, metric)
            assert 0.0 <= value <= 1.0


def test_the_baseline_itself_passes() -> None:
    rows, problems = compare_to_baseline(gated_results(BASELINE), BASELINE)
    assert problems == []
    assert rows
    assert all(r.ok for r in rows)
    assert "gate passed" in gate_report(rows, problems)


@pytest.mark.parametrize(("drop", "ok"), [(0.0, True), (GATE_TOLERANCE, True), (0.03, False)])
def test_a_metric_may_drop_by_the_tolerance_and_no_more(drop: float, ok: bool) -> None:
    results = gated_results(BASELINE)
    entry = results["categories", "public_sample"]
    accuracy = entry["categories"]["accuracy"]
    _set(entry, "categories.accuracy", accuracy - drop)
    rows, problems = compare_to_baseline(results, BASELINE)
    row = next(r for r in rows if r.metric == "categories.accuracy")
    assert row.ok is ok
    assert problems == []
    report = gate_report(rows, problems)
    assert ("gate passed" in report) is ok
    assert ("DROPPED" in report) is not ok


def test_a_metric_that_rises_passes() -> None:
    results = gated_results(BASELINE)
    _set(results["bursts", "gate_generated"], "pairs.precision", 0.99)
    rows, _ = compare_to_baseline(results, BASELINE)
    assert all(r.ok for r in rows)


def test_a_missing_metric_fails() -> None:
    results = gated_results(BASELINE)
    _set(results["dedupe", "gate_generated"], "dedupe.precision", None)
    rows, _ = compare_to_baseline(results, BASELINE)
    assert [r.metric for r in rows if not r.ok] == ["dedupe.precision"]


def test_a_changed_sample_or_a_missing_baseline_is_a_problem() -> None:
    results = gated_results(BASELINE)
    results["categories", "public_sample"]["photos"] += 1
    baseline = copy.deepcopy(BASELINE)
    del baseline["bursts"]["gate_generated"]
    rows, problems = compare_to_baseline(results, baseline)
    assert any("public_sample: 26 photos, the baseline has 25" in p for p in problems)
    assert "bursts.gate_generated: no baseline" in problems
    assert "gate FAILED: 0 metrics dropped, 2 problems" in gate_report(rows, problems)


def test_the_gate_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = gated_results(BASELINE)
    monkeypatch.setattr(run_eval, "evaluate_gate", lambda work: results)
    assert run_eval.main(["gate"]) == 0
    assert "gate passed" in capsys.readouterr().out

    _set(results["dedupe", "gate_generated"], "dedupe.recall", 0.5)
    assert run_eval.main(["gate", "--work", str(tmp_path)]) == 1
    assert "dedupe.recall" in capsys.readouterr().out

    saved = tmp_path / "baseline.json"
    monkeypatch.setattr(run_eval, "BASELINE", saved)
    assert run_eval.main(["gate", "--write-baseline"]) == 0
    written = json.loads(saved.read_text(encoding="utf-8"))
    assert written["dedupe"]["gate_generated"]["dedupe"]["recall"] == 0.5
    assert set(written) == {"categories", "dedupe", "bursts"}
