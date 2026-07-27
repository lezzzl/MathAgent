"""Prepare artifact, evaluation, and pairwise data for the dashboard UI."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dashboard.artifacts import (
    RunArtifact,
    evaluation_cache_path,
    file_sha256,
    load_task_records,
)
from dashboard.evaluation import grader_version, load_evaluations
from dashboard.statistics import (
    PairwiseResult,
    adjust_results,
    compare_paired_scores,
)


EVALUATION_COLUMNS = [
    "run_id",
    "benchmark_name",
    "task_id",
    "status",
    "score",
    "extracted_prediction",
    "extracted_ground_truth",
    "reason",
]


def load_available_evaluations(
    runs: dict[str, RunArtifact], evaluations_dir: Path
) -> pd.DataFrame:
    """Load only sidecars matching current artifact content and grader version."""

    rows: list[dict[str, Any]] = []
    current_version = grader_version()
    for run in runs.values():
        for artifact in run.benchmarks.values():
            fingerprint = file_sha256(artifact.path)
            path = evaluation_cache_path(
                evaluations_dir, artifact, fingerprint, current_version
            )
            if not path.is_file():
                continue
            for record in load_evaluations(
                path,
                artifact=artifact,
                source_fingerprint=fingerprint,
                version_string=current_version,
            ):
                rows.append(
                    {
                        column: getattr(record, column)
                        for column in EVALUATION_COLUMNS
                    }
                )
    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)


def manifest_rows(runs: dict[str, RunArtifact]) -> pd.DataFrame:
    """Flatten stable run configuration and operational metrics."""

    rows: list[dict[str, Any]] = []
    for run in runs.values():
        manifest = run.manifest
        generation = manifest.get("generation") or {}
        for benchmark_name, artifact in run.benchmarks.items():
            entry = artifact.manifest_entry
            summary = entry.get("summary") or {}
            rows.append(
                {
                    "run_id": run.run_id,
                    "model": manifest.get("model"),
                    "prompt_version": manifest.get("prompt_version"),
                    "status": entry.get("status"),
                    "benchmark_name": benchmark_name,
                    "tasks": summary.get("total_tasks", entry.get("total_tasks")),
                    "successful_calls": summary.get("successful_tasks"),
                    "failed_calls": summary.get("failed_tasks"),
                    "input_tokens": summary.get("input_tokens"),
                    "output_tokens": summary.get("output_tokens"),
                    "total_tokens": summary.get("total_tokens"),
                    "wall_time_seconds": summary.get("wall_time_seconds"),
                    "tasks_per_second": summary.get("tasks_per_second"),
                    "temperature": generation.get("temperature"),
                    "seed": generation.get("seed"),
                    "thinking": generation.get("thinking"),
                    "max_tokens": generation.get("max_tokens"),
                }
            )
    return pd.DataFrame(rows)


def accuracy_rows(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Summarize correctness while keeping unresolved grades visible."""

    if evaluations.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "benchmark_name",
                "scored",
                "correct",
                "incorrect",
                "unresolved",
                "accuracy",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (run_id, benchmark), group in evaluations.groupby(
        ["run_id", "benchmark_name"], sort=True
    ):
        scored = group["score"].notna()
        correct = int((group.loc[scored, "score"] == True).sum())  # noqa: E712
        scored_count = int(scored.sum())
        rows.append(
            {
                "run_id": run_id,
                "benchmark_name": benchmark,
                "scored": scored_count,
                "correct": correct,
                "incorrect": scored_count - correct,
                "unresolved": int((~scored).sum()),
                "accuracy": correct / scored_count if scored_count else None,
            }
        )
    return pd.DataFrame(rows)


def _score_map(
    evaluations: pd.DataFrame, run_id: str, benchmarks: Iterable[str]
) -> dict[tuple[str, str], bool | None]:
    selected = evaluations[
        (evaluations["run_id"] == run_id)
        & evaluations["benchmark_name"].isin(list(benchmarks))
    ]
    return {
        (str(row.benchmark_name), str(row.task_id)): (
            None if pd.isna(row.score) else bool(row.score)
        )
        for row in selected.itertuples()
    }


def pairwise_results(
    evaluations: pd.DataFrame,
    run_ids: list[str],
    benchmarks: list[str],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> list[PairwiseResult]:
    """Build a Holm-adjusted displayed family across runs and scopes."""

    results: list[PairwiseResult] = []
    for left_run, right_run in combinations(sorted(run_ids), 2):
        for benchmark in sorted(benchmarks):
            left = _score_map(evaluations, left_run, [benchmark])
            right = _score_map(evaluations, right_run, [benchmark])
            results.append(
                compare_paired_scores(
                    left_run,
                    right_run,
                    left,
                    right,
                    scope=benchmark,
                    weighting="task",
                    n_resamples=n_resamples,
                    seed=seed,
                )
            )
    return adjust_results(results)


def pairwise_frame(results: list[PairwiseResult]) -> pd.DataFrame:
    """Convert comparison results to user-facing percentage-point columns."""

    rows = []
    for result in results:
        rows.append(
            {
                "left_run": result.left_run,
                "right_run": result.right_run,
                "scope": result.scope,
                "paired_tasks": result.paired_tasks,
                "unresolved": result.unresolved_tasks,
                "left_accuracy": _percent(result.left_accuracy),
                "right_accuracy": _percent(result.right_accuracy),
                "delta_pp": _percent(result.accuracy_delta),
                "ci_low_pp": _percent(result.ci_low),
                "ci_high_pp": _percent(result.ci_high),
                "left_wins": result.left_wins,
                "right_wins": result.right_wins,
                "ties": result.ties,
                "p_value": result.p_value,
                "adjusted_p_value": result.adjusted_p_value,
            }
        )
    return pd.DataFrame(rows)


def pairwise_display_tables(
    results: list[PairwiseResult],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Build the compact, dynamically named table shown for each run pair."""

    grouped: dict[tuple[str, str], list[PairwiseResult]] = {}
    for result in results:
        grouped.setdefault((result.left_run, result.right_run), []).append(result)

    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for (left_run, right_run), pair_results in grouped.items():
        diff_column = f"Diff ({right_run} − {left_run})"
        rows: list[dict[str, Any]] = []
        for result in pair_results:
            rows.append(
                {
                    "Benchmark": result.scope,
                    left_run: _percent(result.left_accuracy),
                    right_run: _percent(result.right_accuracy),
                    diff_column: _percent(
                        None
                        if result.accuracy_delta is None
                        else -result.accuracy_delta
                    ),
                    "p-value": result.adjusted_p_value,
                }
            )
        tables[(left_run, right_run)] = pd.DataFrame(rows)
    return tables


def p_value_style(p_value: float | None, difference: float | None) -> str:
    """Return directional significance coloring for one p-value cell."""

    if (
        p_value is None
        or difference is None
        or pd.isna(p_value)
        or pd.isna(difference)
        or p_value >= 0.05
        or difference == 0
    ):
        return ""
    strong = p_value < 0.01
    if difference > 0:
        background = "#b7e4c7" if strong else "#d8f3dc"
        foreground = "#14532d"
    else:
        background = "#f4b6b6" if strong else "#fde2e2"
        foreground = "#7f1d1d"
    return (
        f"background-color: {background}; color: {foreground}; "
        "font-weight: 600"
    )


def _percent(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def task_rows(run: RunArtifact, benchmark_name: str) -> pd.DataFrame:
    """Lazily load full task text for one selected run and benchmark."""

    artifact = run.benchmarks[benchmark_name]
    rows: list[dict[str, Any]] = []
    for record in load_task_records(artifact.path):
        metadata = record.get("metadata") or {}
        usage = metadata.get("usage") or {}
        rows.append(
            {
                "run_id": run.run_id,
                "benchmark_name": benchmark_name,
                "task_id": str(record["task_id"]),
                "solution": record.get("solution"),
                "reasoning": record.get("reasoning"),
                "ground_truth": record.get("ground_truth"),
                "latency_seconds": metadata.get("latency_seconds"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "error": metadata.get("error"),
                "category": metadata.get("Category"),
                "subcategory": metadata.get("Subcategory"),
            }
        )
    return pd.DataFrame(rows)
