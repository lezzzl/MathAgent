"""Composable Kedro pipelines for the benchmark experiment lifecycle."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_agent_run,
    evaluate_selected_run,
    run_agentic_loop,
    update_comparison_table,
)


def _agent_node() -> object:
    return node(
        func=run_agentic_loop,
        inputs="params:experiment",
        outputs="agent_run",
        name="run_agentic_loop",
        tags=["agent", "stage_1"],
    )


def _evaluate_node(*, chained: bool) -> object:
    return node(
        func=evaluate_agent_run if chained else evaluate_selected_run,
        inputs=(
            ["agent_run", "params:experiment"]
            if chained
            else "params:experiment"
        ),
        outputs="evaluated_run",
        name="evaluate_results",
        tags=["evaluate", "stage_2"],
    )


def _compare_node(*, chained: bool) -> object:
    return node(
        func=update_comparison_table,
        inputs=(
            ["params:experiment", "evaluated_run"]
            if chained
            else "params:experiment"
        ),
        outputs="comparison_update",
        name="update_comparison_table",
        tags=["compare", "stage_3"],
    )


def create_agent_pipeline() -> Pipeline:
    """Stage 1 only."""

    return pipeline([_agent_node()])


def create_evaluate_pipeline() -> Pipeline:
    """Stage 2 only, resolving the run from configuration."""

    return pipeline([_evaluate_node(chained=False)])


def create_compare_pipeline() -> Pipeline:
    """Stage 3 only."""

    return pipeline([_compare_node(chained=False)])


def create_evaluate_compare_pipeline() -> Pipeline:
    """Stages 2 and 3, useful when agent outputs already exist."""

    return pipeline(
        [
            _evaluate_node(chained=False),
            _compare_node(chained=True),
        ]
    )


def create_experiment_pipeline() -> Pipeline:
    """Stages 1, 2, and 3 with explicit data dependencies."""

    return pipeline(
        [
            _agent_node(),
            _evaluate_node(chained=True),
            _compare_node(chained=True),
        ]
    )
