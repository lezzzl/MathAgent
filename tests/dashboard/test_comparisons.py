from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.artifacts import ArtifactError, RunArtifact, load_run
from dashboard.compare import main as compare_main
from dashboard.comparisons import (
    add_new_runs_to_comparison_table,
    add_run_to_comparison_table,
    load_comparison_table,
)


def _write_run(
    root: Path,
    run_id: str,
    scores: list[object],
    *,
    score_key: str = "score",
) -> RunArtifact:
    directory = root / run_id
    directory.mkdir(parents=True)
    output = directory / "bench.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for index, score in enumerate(scores):
            record = {
                "run_id": run_id,
                "benchmark_name": "Bench",
                "task_id": str(index),
                "solution": "already graded",
                score_key: score,
            }
            stream.write(json.dumps(record) + "\n")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "benchmarks": {"Bench": {"output": str(output)}},
            }
        ),
        encoding="utf-8",
    )
    return load_run(directory)


def test_add_run_builds_and_persists_comparisons_from_precomputed_scores(
    tmp_path: Path,
) -> None:
    left = _write_run(tmp_path, "run-a", [True, False, True], score_key="is_correct")
    right = _write_run(tmp_path, "run-b", [0.0, 0.0, 1.0])
    path = tmp_path / "comparisons.parquet"

    table = add_run_to_comparison_table(
        left,
        {left.run_id: left, right.run_id: right},
        path,
        n_resamples=20,
    )

    assert path.is_file()
    assert len(table) == 1
    row = table.iloc[0]
    assert row["benchmark_name"] == "Bench"
    assert row["left_run"] == "run-a"
    assert row["right_run"] == "run-b"
    assert row["left_avg_score"] == pytest.approx(2 / 3)
    assert row["right_avg_score"] == pytest.approx(1 / 3)
    assert row["diff"] == pytest.approx(-1 / 3)
    assert row["paired_tasks"] == 3
    pd.testing.assert_frame_equal(table, load_comparison_table(path))


def test_adding_existing_run_replaces_its_rows_instead_of_duplicating(
    tmp_path: Path,
) -> None:
    left = _write_run(tmp_path, "run-a", [True, False])
    right = _write_run(tmp_path, "run-b", [False, False])
    path = tmp_path / "comparisons.parquet"
    runs = {left.run_id: left, right.run_id: right}
    add_run_to_comparison_table(left, runs, path, n_resamples=10)

    output = right.benchmarks["Bench"].path
    records = [
        {
            "run_id": "run-b",
            "benchmark_name": "Bench",
            "task_id": str(index),
            "score": True,
        }
        for index in range(2)
    ]
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    refreshed_right = load_run(right.directory)
    table = add_run_to_comparison_table(
        refreshed_right,
        {left.run_id: left, right.run_id: refreshed_right},
        path,
        n_resamples=10,
    )

    assert len(table) == 1
    assert table.iloc[0]["right_avg_score"] == pytest.approx(1.0)


def test_add_run_rejects_artifact_without_precomputed_scores(tmp_path: Path) -> None:
    unscored = _write_run(tmp_path, "unscored", [None, None])

    with pytest.raises(ArtifactError, match="no precomputed"):
        add_run_to_comparison_table(
            unscored,
            {unscored.run_id: unscored},
            tmp_path / "comparisons.parquet",
        )


def test_invalid_precomputed_score_is_rejected(tmp_path: Path) -> None:
    invalid = _write_run(tmp_path, "invalid", [1.5])

    with pytest.raises(ArtifactError, match="Invalid precomputed score"):
        add_run_to_comparison_table(
            invalid,
            {invalid.run_id: invalid},
            tmp_path / "comparisons.parquet",
        )


def test_incremental_update_only_computes_pairs_involving_new_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    left = _write_run(runs_root, "run-a", [True, False])
    right = _write_run(runs_root, "run-b", [False, False])
    output = tmp_path / "comparisons.parquet"

    initial = add_new_runs_to_comparison_table(
        {left.run_id: left, right.run_id: right}, output
    )
    assert initial.added_runs == ("run-a", "run-b")
    assert initial.comparison_rows_added == 1
    old_pair = initial.table.iloc[0].copy()

    left_output = left.benchmarks["Bench"].path
    changed = [
        {
            "run_id": "run-a",
            "benchmark_name": "Bench",
            "task_id": str(index),
            "score": True,
        }
        for index in range(2)
    ]
    left_output.write_text(
        "".join(json.dumps(record) + "\n" for record in changed),
        encoding="utf-8",
    )
    new = _write_run(runs_root, "run-c", [True, True])
    runs = {left.run_id: left, right.run_id: right, new.run_id: new}

    incremental = add_new_runs_to_comparison_table(runs, output)

    assert incremental.added_runs == ("run-c",)
    assert incremental.comparison_rows_added == 2
    assert len(incremental.table) == 3
    preserved = incremental.table[
        (incremental.table["left_run"] == "run-a")
        & (incremental.table["right_run"] == "run-b")
    ].iloc[0]
    assert preserved["left_avg_score"] == old_pair["left_avg_score"]
    assert preserved["updated_at"] == old_pair["updated_at"]

    unchanged = add_new_runs_to_comparison_table(runs, output)
    assert unchanged.added_runs == ()
    assert unchanged.comparison_rows_added == 0


def test_compare_cli_creates_incremental_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "run-a", [True])
    _write_run(runs_root, "run-b", [False])
    output = tmp_path / "comparisons.parquet"

    result = compare_main(
        ["--runs-dir", str(runs_root), "--output", str(output)]
    )

    assert result == 0
    assert output.is_file()
    assert "added comparison rows: 1" in capsys.readouterr().out
