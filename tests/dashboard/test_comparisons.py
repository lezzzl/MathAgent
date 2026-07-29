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
    write_comparison_table,
)
from dashboard.statistics import holm_adjust


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


def _write_multi_benchmark_run(
    root: Path,
    run_id: str,
    scores_by_benchmark: dict[str, list[object]],
) -> RunArtifact:
    directory = root / run_id
    directory.mkdir(parents=True)
    benchmarks: dict[str, dict[str, str]] = {}
    for benchmark_name, scores in scores_by_benchmark.items():
        output = directory / f"{benchmark_name.lower()}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for index, score in enumerate(scores):
                stream.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "benchmark_name": benchmark_name,
                            "task_id": str(index),
                            "solution": "already graded",
                            "score": score,
                        }
                    )
                    + "\n"
                )
        benchmarks[benchmark_name] = {"output": str(output)}
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "benchmarks": benchmarks}),
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


def test_update_preserves_existing_pairs_when_adding_new_run(
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


def test_update_creates_all_pairs_when_table_is_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs = {
        run.run_id: run
        for run in (
            _write_run(runs_root, "run-a", [True, False]),
            _write_run(runs_root, "run-b", [False, False]),
            _write_run(runs_root, "run-c", [True, True]),
        )
    }
    output = tmp_path / "comparisons.parquet"

    update = add_new_runs_to_comparison_table(runs, output, n_resamples=10)

    assert output.is_file()
    assert update.added_runs == ("run-a", "run-b", "run-c")
    assert update.comparison_rows_added == 3
    assert {
        (row.left_run, row.right_run)
        for row in update.table.itertuples(index=False)
    } == {("run-a", "run-b"), ("run-a", "run-c"), ("run-b", "run-c")}


def test_update_creates_empty_table_and_reports_unscored_runs(
    tmp_path: Path,
) -> None:
    unscored = _write_run(tmp_path / "runs", "run-a", [None, None])
    output = tmp_path / "comparisons.parquet"

    update = add_new_runs_to_comparison_table(
        {unscored.run_id: unscored}, output
    )

    assert output.is_file()
    assert update.table.empty
    assert update.added_runs == ()
    assert update.skipped_runs == ("run-a",)
    pd.testing.assert_frame_equal(update.table, load_comparison_table(output))


def test_update_fills_missing_pair_when_every_run_is_represented(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    left = _write_run(runs_root, "run-a", [True, True, False])
    middle = _write_run(runs_root, "run-b", [False, False, False])
    right = _write_run(runs_root, "run-c", [True, False, True])
    runs = {run.run_id: run for run in (left, middle, right)}
    output = tmp_path / "comparisons.parquet"
    partial = add_run_to_comparison_table(
        middle, runs, output, n_resamples=10
    )
    partial["adjusted_p_value"] = 0.123
    write_comparison_table(output, partial)
    preserved_columns = [
        column
        for column in partial.columns
        if column != "adjusted_p_value"
    ]

    update = add_new_runs_to_comparison_table(runs, output, n_resamples=10)

    assert update.added_runs == ()
    assert update.comparison_rows_added == 1
    assert len(update.table) == 3
    old_pairs = update.table[
        (update.table["left_run"] == "run-b")
        | (update.table["right_run"] == "run-b")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        old_pairs[preserved_columns],
        partial[preserved_columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert update.table["adjusted_p_value"].tolist() == pytest.approx(
        holm_adjust(update.table["p_value"].astype(float).tolist())
    )


def test_update_adds_missing_benchmark_row_for_existing_pair(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    left = _write_multi_benchmark_run(
        runs_root,
        "run-a",
        {"BenchA": [True, False], "BenchB": [True, True]},
    )
    right = _write_multi_benchmark_run(
        runs_root,
        "run-b",
        {"BenchA": [False, False], "BenchB": [True, False]},
    )
    runs = {left.run_id: left, right.run_id: right}
    output = tmp_path / "comparisons.parquet"
    complete = add_new_runs_to_comparison_table(runs, output, n_resamples=10)
    partial = complete.table[
        complete.table["benchmark_name"] == "BenchA"
    ].copy()
    write_comparison_table(output, partial)

    update = add_new_runs_to_comparison_table(runs, output, n_resamples=10)

    assert update.added_runs == ()
    assert update.comparison_rows_added == 1
    assert update.table["benchmark_name"].tolist() == ["BenchA", "BenchB"]
    assert update.table.iloc[0]["updated_at"] == partial.iloc[0]["updated_at"]


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
