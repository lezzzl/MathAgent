"""Shape persistent comparison rows for the dashboard views."""

from __future__ import annotations

from typing import Any

import pandas as pd


def comparison_run_ids(comparisons: pd.DataFrame) -> list[str]:
    """Return every run represented in at least one comparison."""

    if comparisons.empty:
        return []
    return sorted(
        set(comparisons["left_run"].astype(str))
        | set(comparisons["right_run"].astype(str))
    )


def _display_p_value(row: pd.Series) -> float | None:
    adjusted = row.get("adjusted_p_value")
    if adjusted is not None and not pd.isna(adjusted):
        return float(adjusted)
    raw = row.get("p_value")
    return None if raw is None or pd.isna(raw) else float(raw)


def _oriented_rows(
    comparisons: pd.DataFrame, left_run: str, right_run: str
) -> list[dict[str, Any]]:
    """Orient canonical stored pairs to the run order requested by the UI."""

    canonical_left, canonical_right = sorted((left_run, right_run))
    selected = comparisons[
        (comparisons["left_run"] == canonical_left)
        & (comparisons["right_run"] == canonical_right)
    ]
    rows: list[dict[str, Any]] = []
    requested_is_canonical = left_run == canonical_left
    for row in selected.sort_values("benchmark_name").to_dict("records"):
        if requested_is_canonical:
            left_score = row["left_avg_score"]
            right_score = row["right_avg_score"]
            difference = row["diff"]
        else:
            left_score = row["right_avg_score"]
            right_score = row["left_avg_score"]
            difference = -row["diff"]
        rows.append(
            {
                "benchmark_name": row["benchmark_name"],
                "left_score": left_score,
                "right_score": right_score,
                "difference": difference,
                "p_value": _display_p_value(pd.Series(row)),
            }
        )
    return rows


def two_run_comparison_table(
    comparisons: pd.DataFrame, left_run: str, right_run: str
) -> pd.DataFrame:
    """Build the five-column side-by-side comparison table."""

    rows = _oriented_rows(comparisons, left_run, right_run)
    return pd.DataFrame(
        [
            {
                "Benchmark": row["benchmark_name"],
                left_run: _percent(row["left_score"]),
                right_run: _percent(row["right_score"]),
                "Diff": _percent(row["difference"]),
                "p-value": row["p_value"],
            }
            for row in rows
        ],
        columns=["Benchmark", left_run, right_run, "Diff", "p-value"],
    )


def baseline_comparison_table(
    comparisons: pd.DataFrame,
    baseline_run: str,
    compared_runs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build baseline scores, comparison deltas, and a parallel p-value matrix."""

    oriented = {
        run_id: {
            str(row["benchmark_name"]): row
            for row in _oriented_rows(comparisons, baseline_run, run_id)
        }
        for run_id in compared_runs
        if run_id != baseline_run
    }
    benchmarks = sorted(
        {
            benchmark
            for rows_by_benchmark in oriented.values()
            for benchmark in rows_by_benchmark
        }
    )
    display_rows: list[dict[str, Any]] = []
    p_value_rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        available = [
            rows_by_benchmark[benchmark]
            for rows_by_benchmark in oriented.values()
            if benchmark in rows_by_benchmark
        ]
        baseline_score = available[0]["left_score"] if available else None
        display_row: dict[str, Any] = {
            "Benchmark": benchmark,
            baseline_run: _percent(baseline_score),
        }
        p_value_row: dict[str, Any] = {"Benchmark": benchmark}
        for run_id in compared_runs:
            row = oriented.get(run_id, {}).get(benchmark)
            display_row[run_id] = (
                None if row is None else _percent(row["difference"])
            )
            p_value_row[run_id] = None if row is None else row["p_value"]
        display_rows.append(display_row)
        p_value_rows.append(p_value_row)

    columns = ["Benchmark", baseline_run, *compared_runs]
    p_value_columns = ["Benchmark", *compared_runs]
    return (
        pd.DataFrame(display_rows, columns=columns),
        pd.DataFrame(p_value_rows, columns=p_value_columns),
    )


def p_value_style(p_value: float | None, difference: float | None) -> str:
    """Return directional significance coloring based on a p-value."""

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
    return None if value is None or pd.isna(value) else 100.0 * float(value)
