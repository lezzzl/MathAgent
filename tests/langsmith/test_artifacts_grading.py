from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_io import ArtifactError, load_local_run
from grading import grade_answer


def _write_run(root: Path, rows: list[dict]) -> None:
    run = root / "run-a"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "benchmarks": {"Bench": {"output": "bench.jsonl"}},
            }
        ),
        encoding="utf-8",
    )
    (run / "bench.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_local_run_validates_and_normalizes(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        [
            {
                "benchmark_name": "Bench",
                "task_id": 7,
                "problem": "1+1",
                "ground_truth": "2",
                "solution": "2",
                "metadata": {"split": "test"},
            }
        ],
    )
    run = load_local_run(tmp_path, "run-a")
    assert run.tasks[0].example_key == "Bench:7"
    assert run.tasks[0].problem == "1+1"


def test_duplicate_task_is_rejected(tmp_path: Path) -> None:
    row = {"benchmark_name": "Bench", "task_id": "1"}
    _write_run(tmp_path, [row, row])
    with pytest.raises(ArtifactError, match="Duplicate task"):
        load_local_run(tmp_path, "run-a")


def test_grading_distinguishes_incorrect_and_unresolved() -> None:
    parse = lambda value: [value]
    verify = lambda expected, actual: expected == actual
    assert grade_answer("", "2", parse=parse, verify=verify).correct is False
    assert grade_answer("2", None, parse=parse, verify=verify).correct is None
    assert grade_answer("2", "2", parse=parse, verify=verify).correct is True
