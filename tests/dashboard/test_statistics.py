from __future__ import annotations

import pandas as pd
import pytest

from dashboard.data import pairwise_display_tables, pairwise_results, p_value_style
from dashboard.statistics import PairwiseResult, compare_paired_scores, holm_adjust


def test_paired_comparison_aligns_by_task_id_and_counts_unresolved() -> None:
    left = {
        ("A", "1"): True,
        ("A", "2"): False,
        ("A", "3"): None,
        ("A", "left-only"): True,
    }
    right = {
        ("A", "2"): True,
        ("A", "1"): False,
        ("A", "3"): True,
        ("A", "right-only"): True,
    }

    result = compare_paired_scores(
        "left", "right", left, right, scope="A", n_resamples=200, seed=7
    )

    assert result.paired_tasks == 2
    assert result.unresolved_tasks == 1
    assert result.left_wins == 1
    assert result.right_wins == 1
    assert result.ties == 0
    assert result.accuracy_delta == pytest.approx(0)
    assert result.p_value == pytest.approx(1)


def test_all_ties_have_point_interval_and_p_one() -> None:
    scores = {("A", str(index)): index % 2 == 0 for index in range(10)}

    result = compare_paired_scores(
        "left", "right", scores, scores, scope="A", n_resamples=100
    )

    assert result.accuracy_delta == 0
    assert result.ci_low == 0
    assert result.ci_high == 0
    assert result.p_value == 1
    assert result.ties == 10


def test_empty_overlap_has_no_inference() -> None:
    result = compare_paired_scores(
        "left",
        "right",
        {("A", "1"): True},
        {("A", "2"): True},
        scope="A",
        n_resamples=10,
    )

    assert result.paired_tasks == 0
    assert result.accuracy_delta is None
    assert result.p_value is None


def test_bootstrap_is_reproducible() -> None:
    left = {("A", str(index)): index < 7 for index in range(10)}
    right = {("A", str(index)): index < 4 for index in range(10)}

    first = compare_paired_scores(
        "left", "right", left, right, scope="A", n_resamples=500, seed=123
    )
    second = compare_paired_scores(
        "left", "right", left, right, scope="A", n_resamples=500, seed=123
    )

    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_benchmark_weighting_does_not_let_large_benchmark_dominate() -> None:
    left = {("large", str(index)): False for index in range(10)}
    right = {("large", str(index)): True for index in range(10)}
    left[("small", "1")] = True
    right[("small", "1")] = False

    task_weighted = compare_paired_scores(
        "left", "right", left, right, scope="all", weighting="task", n_resamples=20
    )
    benchmark_weighted = compare_paired_scores(
        "left",
        "right",
        left,
        right,
        scope="all",
        weighting="benchmark",
        n_resamples=20,
    )

    assert task_weighted.accuracy_delta == pytest.approx(-9 / 11)
    assert benchmark_weighted.accuracy_delta == pytest.approx(0)


def test_holm_adjustment_is_monotonic_in_sorted_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, None])
    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert adjusted[3] is None


def test_pairwise_display_table_uses_third_minus_second_difference() -> None:
    result = PairwiseResult(
        left_run="run-a",
        right_run="run-b",
        scope="AIME",
        weighting="task",
        paired_tasks=30,
        unresolved_tasks=0,
        left_accuracy=0.7,
        right_accuracy=0.8,
        accuracy_delta=-0.1,
        ci_low=-0.2,
        ci_high=0,
        left_wins=1,
        right_wins=4,
        ties=25,
        p_value=0.03,
        adjusted_p_value=0.04,
    )

    table = pairwise_display_tables([result])[("run-a", "run-b")]

    assert list(table.columns) == [
        "Benchmark",
        "run-a",
        "run-b",
        "Diff (run-b − run-a)",
        "p-value",
    ]
    assert table.iloc[0].to_dict() == {
        "Benchmark": "AIME",
        "run-a": pytest.approx(70),
        "run-b": pytest.approx(80),
        "Diff (run-b − run-a)": pytest.approx(10),
        "p-value": pytest.approx(0.04),
    }


def test_p_value_coloring_uses_significance_strength_and_direction() -> None:
    assert "#b7e4c7" in p_value_style(0.001, 5)
    assert "#d8f3dc" in p_value_style(0.03, 5)
    assert "#f4b6b6" in p_value_style(0.001, -5)
    assert "#fde2e2" in p_value_style(0.03, -5)
    assert p_value_style(0.05, 5) == ""
    assert p_value_style(0.001, 0) == ""


def test_pairwise_results_contains_benchmarks_without_all_rows() -> None:
    evaluations = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "benchmark_name": benchmark,
                "task_id": "1",
                "score": score,
            }
            for run_id, score in (("run-a", True), ("run-b", False))
            for benchmark in ("AIME", "HMMT")
        ]
    )

    results = pairwise_results(
        evaluations,
        ["run-a", "run-b"],
        ["AIME", "HMMT"],
        n_resamples=10,
    )

    assert [result.scope for result in results] == ["AIME", "HMMT"]
