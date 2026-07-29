"""Persistent pairwise comparisons built from pre-scored run artifacts."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dashboard.artifacts import (
    ArtifactError,
    RunArtifact,
    file_sha256,
    load_task_records,
)
from dashboard.statistics import compare_paired_scores, holm_adjust


COMPARISON_COLUMNS = [
    "benchmark_name",
    "left_run",
    "right_run",
    "left_avg_score",
    "right_avg_score",
    "diff",
    "paired_tasks",
    "unresolved_tasks",
    "left_wins",
    "right_wins",
    "ties",
    "p_value",
    "adjusted_p_value",
    "left_source_fingerprint",
    "right_source_fingerprint",
    "updated_at",
]

_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class IncrementalUpdate:
    """Result of adding comparisons not yet represented in the table."""

    table: pd.DataFrame
    added_runs: tuple[str, ...]
    skipped_runs: tuple[str, ...]
    comparison_rows_added: int


def empty_comparison_table() -> pd.DataFrame:
    """Return an empty comparison table with the stable storage schema."""

    return pd.DataFrame(columns=COMPARISON_COLUMNS)


def load_comparison_table(path: Path) -> pd.DataFrame:
    """Load and validate the persistent comparison table."""

    if not path.is_file():
        return empty_comparison_table()
    try:
        table = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Cannot read comparison table {path}: {exc}") from exc

    missing = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing:
        raise ArtifactError(
            f"Comparison table {path} is missing columns: {', '.join(sorted(missing))}"
        )
    table = table[COMPARISON_COLUMNS].copy()
    duplicate = table.duplicated(
        subset=["benchmark_name", "left_run", "right_run"], keep=False
    )
    if duplicate.any():
        raise ArtifactError(f"Duplicate run-pair rows in comparison table {path}")
    invalid_order = table["left_run"].astype(str) >= table["right_run"].astype(str)
    if invalid_order.any():
        raise ArtifactError(
            f"Comparison table {path} contains non-canonical run pairs"
        )
    return table.sort_values(
        ["benchmark_name", "left_run", "right_run"], ignore_index=True
    )


def write_comparison_table(path: Path, table: pd.DataFrame) -> None:
    """Atomically persist a validated comparison table."""

    missing = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing:
        raise ArtifactError(
            f"Cannot write comparison table; missing: {', '.join(sorted(missing))}"
        )
    ordered = table[COMPARISON_COLUMNS].sort_values(
        ["benchmark_name", "left_run", "right_run"], ignore_index=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with _WRITE_LOCK:
        try:
            ordered.to_parquet(temporary, index=False)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _score_value(record: Mapping[str, Any], path: Path) -> float | None:
    """Read a precomputed score without trying to grade the model output."""

    value = record.get("score")
    if value is None:
        value = record.get("is_correct")
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if math.isfinite(score) and 0.0 <= score <= 1.0:
            return score
    task_id = record.get("task_id", "<unknown>")
    raise ArtifactError(
        f"Invalid precomputed score for task {task_id!r} in {path}; "
        "expected a Boolean or a number from 0 to 1"
    )


def load_run_scores(
    run: RunArtifact,
) -> tuple[dict[str, dict[tuple[str, str], float | None]], dict[str, str]]:
    """Load per-task scores and source fingerprints for one run."""

    scores: dict[str, dict[tuple[str, str], float | None]] = {}
    fingerprints: dict[str, str] = {}
    for benchmark_name, artifact in run.benchmarks.items():
        benchmark_scores: dict[tuple[str, str], float | None] = {}
        for record in load_task_records(artifact.path):
            benchmark_scores[(benchmark_name, str(record["task_id"]))] = _score_value(
                record, artifact.path
            )
        scores[benchmark_name] = benchmark_scores
        fingerprints[benchmark_name] = file_sha256(artifact.path)
    return scores, fingerprints


def _average(scores: Mapping[tuple[str, str], float | None]) -> float | None:
    resolved = [float(value) for value in scores.values() if value is not None]
    return sum(resolved) / len(resolved) if resolved else None


def _has_resolved_score(
    scores: Mapping[str, Mapping[tuple[str, str], float | None]],
) -> bool:
    return any(
        value is not None
        for values in scores.values()
        for value in values.values()
    )


def _comparison_row(
    left_run: str,
    right_run: str,
    benchmark_name: str,
    left_scores: Mapping[tuple[str, str], float | None],
    right_scores: Mapping[tuple[str, str], float | None],
    left_fingerprint: str,
    right_fingerprint: str,
    *,
    n_resamples: int | None,
    seed: int,
) -> dict[str, Any] | None:
    left_average = _average(left_scores)
    right_average = _average(right_scores)
    if left_average is None or right_average is None:
        return None
    result = compare_paired_scores(
        left_run,
        right_run,
        left_scores,
        right_scores,
        scope=benchmark_name,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "benchmark_name": benchmark_name,
        "left_run": left_run,
        "right_run": right_run,
        "left_avg_score": left_average,
        "right_avg_score": right_average,
        "diff": right_average - left_average,
        "paired_tasks": result.paired_tasks,
        "unresolved_tasks": result.unresolved_tasks,
        "left_wins": result.left_wins,
        "right_wins": result.right_wins,
        "ties": result.ties,
        "p_value": result.p_value,
        "adjusted_p_value": float("nan"),
        "left_source_fingerprint": left_fingerprint,
        "right_source_fingerprint": right_fingerprint,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _comparison_rows_for_pair(
    left_run: str,
    right_run: str,
    left_data: tuple[
        dict[str, dict[tuple[str, str], float | None]], dict[str, str]
    ],
    right_data: tuple[
        dict[str, dict[tuple[str, str], float | None]], dict[str, str]
    ],
    *,
    benchmark_names: set[str] | None = None,
    n_resamples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Build all shared-benchmark rows for one canonical run pair."""

    left_scores, left_fingerprints = left_data
    right_scores, right_fingerprints = right_data
    rows: list[dict[str, Any]] = []
    shared_benchmarks = set(left_scores) & set(right_scores)
    if benchmark_names is not None:
        shared_benchmarks &= benchmark_names
    for benchmark_name in sorted(shared_benchmarks):
        row = _comparison_row(
            left_run,
            right_run,
            benchmark_name,
            left_scores[benchmark_name],
            right_scores[benchmark_name],
            left_fingerprints[benchmark_name],
            right_fingerprints[benchmark_name],
            n_resamples=n_resamples,
            seed=seed,
        )
        if row is not None:
            rows.append(row)
    return rows


def _adjust_p_values(table: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm correction across all run pairs within each benchmark."""

    adjusted = table.copy()
    if adjusted.empty:
        return adjusted
    for _, indexes in adjusted.groupby("benchmark_name", sort=False).groups.items():
        index_list = list(indexes)
        values = holm_adjust(
            [
                None if pd.isna(adjusted.at[index, "p_value"]) else float(
                    adjusted.at[index, "p_value"]
                )
                for index in index_list
            ]
        )
        for index, value in zip(index_list, values, strict=True):
            adjusted.at[index, "adjusted_p_value"] = value
    return adjusted


def add_run_to_comparison_table(
    run: RunArtifact,
    runs: Mapping[str, RunArtifact],
    path: Path,
    *,
    n_resamples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Add or refresh one run against every other discovered pre-scored run.

    Existing rows not involving ``run`` are preserved. Rows involving it are
    rebuilt from the current JSONL files, then all per-benchmark Holm-adjusted
    p-values are refreshed before the table is atomically replaced.
    """

    if run.run_id not in runs:
        raise ArtifactError(f"Run {run.run_id!r} is not present in discovered runs")

    run_data = load_run_scores(run)
    run_scores, _ = run_data
    if not _has_resolved_score(run_scores):
        raise ArtifactError(
            f"Run {run.run_id!r} has no precomputed 'score' or 'is_correct' values"
        )

    current = load_comparison_table(path)
    retained = current[
        (current["left_run"] != run.run_id) & (current["right_run"] != run.run_id)
    ].copy()
    new_rows: list[dict[str, Any]] = []

    for other_run_id in sorted(set(runs) - {run.run_id}):
        other = runs[other_run_id]
        other_data = load_run_scores(other)
        other_scores, _ = other_data
        if not _has_resolved_score(other_scores):
            continue

        left_run, right_run = sorted((run.run_id, other_run_id))
        if left_run == run.run_id:
            left_data = run_data
            right_data = other_data
        else:
            left_data = other_data
            right_data = run_data

        new_rows.extend(
            _comparison_rows_for_pair(
                left_run,
                right_run,
                left_data,
                right_data,
                n_resamples=n_resamples,
                seed=seed,
            )
        )

    additions = pd.DataFrame(new_rows, columns=COMPARISON_COLUMNS)
    if retained.empty:
        combined = additions
    elif additions.empty:
        combined = retained
    else:
        combined = pd.concat([retained, additions], ignore_index=True)
    combined = _adjust_p_values(combined)
    write_comparison_table(path, combined)
    return load_comparison_table(path)


def add_new_runs_to_comparison_table(
    runs: Mapping[str, RunArtifact],
    path: Path,
    *,
    n_resamples: int | None = None,
    seed: int = 42,
) -> IncrementalUpdate:
    """Add every missing comparison between discovered pre-scored runs.

    Existing pair statistics are preserved. Every eligible run pair is checked
    for missing shared-benchmark rows, including pairs whose runs are already
    represented elsewhere in the table. Holm-adjusted p-values are refreshed
    when rows are added because the per-benchmark multiple-testing family grows.
    A missing output path is persisted even when there are no compatible pairs.
    """

    table_existed = path.is_file()
    current = load_comparison_table(path)
    represented = (
        set(current["left_run"].astype(str))
        | set(current["right_run"].astype(str))
        if not current.empty
        else set()
    )
    new_run_ids = sorted(set(runs) - represented)

    score_cache: dict[
        str,
        tuple[dict[str, dict[tuple[str, str], float | None]], dict[str, str]],
    ] = {}
    skipped: list[str] = []
    for run_id in sorted(runs):
        run_data = load_run_scores(runs[run_id])
        if not _has_resolved_score(run_data[0]):
            skipped.append(run_id)
            continue
        score_cache[run_id] = run_data

    added_run_ids = sorted(set(new_run_ids) & set(score_cache))
    existing_keys = set(
        zip(
            current["benchmark_name"].astype(str),
            current["left_run"].astype(str),
            current["right_run"].astype(str),
            strict=True,
        )
    )
    new_rows: list[dict[str, Any]] = []
    for left_run, right_run in combinations(sorted(score_cache), 2):
        shared_benchmarks = (
            set(score_cache[left_run][0]) & set(score_cache[right_run][0])
        )
        missing_benchmarks = {
            benchmark_name
            for benchmark_name in shared_benchmarks
            if (benchmark_name, left_run, right_run) not in existing_keys
        }
        if not missing_benchmarks:
            continue
        new_rows.extend(
            _comparison_rows_for_pair(
                left_run,
                right_run,
                score_cache[left_run],
                score_cache[right_run],
                benchmark_names=missing_benchmarks,
                n_resamples=n_resamples,
                seed=seed,
            )
        )

    if not new_rows:
        if not table_existed:
            write_comparison_table(path, current)
            current = load_comparison_table(path)
        return IncrementalUpdate(
            current,
            tuple(added_run_ids),
            tuple(sorted(skipped)),
            0,
        )

    additions = pd.DataFrame(new_rows, columns=COMPARISON_COLUMNS)
    combined = (
        additions
        if current.empty
        else pd.concat([current, additions], ignore_index=True)
    )
    combined = _adjust_p_values(combined)
    write_comparison_table(path, combined)
    stored = load_comparison_table(path)
    return IncrementalUpdate(
        stored,
        tuple(added_run_ids),
        tuple(sorted(skipped)),
        len(additions),
    )
