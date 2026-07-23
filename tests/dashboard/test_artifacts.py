from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.artifacts import (
    ArtifactError,
    discover_runs,
    file_sha256,
    load_task_records,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def test_discover_runs_relocates_machine_specific_output_path(tmp_path: Path) -> None:
    run = tmp_path / "run-one"
    run.mkdir()
    output = run / "aime.jsonl"
    _write_jsonl(
        output,
        [
            {
                "run_id": "run-one",
                "benchmark_name": "AIME",
                "task_id": "2",
                "solution": "2",
                "ground_truth": "2",
            }
        ],
    )
    _write_json(
        run / "manifest.json",
        {
            "run_id": "run-one",
            "benchmarks": {
                "AIME": {"output": "/another/machine/results/run-one/aime.jsonl"}
            },
        },
    )

    runs = discover_runs(tmp_path)

    assert list(runs) == ["run-one"]
    assert runs["run-one"].benchmarks["AIME"].path == output


def test_task_records_reject_duplicate_identity(tmp_path: Path) -> None:
    output = tmp_path / "duplicate.jsonl"
    row = {"benchmark_name": "AIME", "task_id": "1"}
    _write_jsonl(output, [row, row])

    with pytest.raises(ArtifactError, match="Duplicate task"):
        load_task_records(output)


def test_fingerprint_changes_when_resumed_file_is_rewritten(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    output.write_text('{"task_id": "1"}\n', encoding="utf-8")
    first = file_sha256(output)
    output.write_text('{"task_id": "1", "solution": "new"}\n', encoding="utf-8")

    assert file_sha256(output) != first


def test_invalid_jsonl_reports_line_number(tmp_path: Path) -> None:
    output = tmp_path / "bad.jsonl"
    output.write_text('{"benchmark_name":"A","task_id":"1"}\n{bad\n', encoding="utf-8")

    with pytest.raises(ArtifactError, match=r"bad\.jsonl:2"):
        load_task_records(output)
