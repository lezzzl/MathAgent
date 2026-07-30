"""Streamlit application for comparing benchmark run artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# This must be set before third-party imports when Streamlit has not already
# loaded PyArrow. The explicit runtime switch below also handles that case.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import pandas as pd
import streamlit as st

from dashboard.artifacts import (
    ArtifactError,
    discover_runs,
    resolve_evaluations_dir,
    resolve_runs_dir,
)
from dashboard.data import (
    accuracy_rows,
    load_available_evaluations,
    manifest_rows,
    pairwise_display_tables,
    pairwise_results,
    p_value_style,
    task_rows,
)
from dashboard.evaluation import evaluate_artifact
from dashboard.runtime import (
    DATAFRAME_SERIALIZATION_LOCK,
    use_system_arrow_memory_pool,
)


use_system_arrow_memory_pool()
st.set_page_config(page_title="MathAgent run comparison", layout="wide")


def _directory_signature(path: Path, patterns: tuple[str, ...]) -> tuple[Any, ...]:
    """Build a cheap cache key that changes when an artifact is rewritten."""

    if not path.exists():
        return (str(path), "missing")
    signature: list[Any] = [str(path)]
    for pattern in patterns:
        for item in sorted(path.glob(pattern)):
            stat = item.stat()
            signature.append((str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def _safe_dataframe(data: Any, **kwargs: Any) -> Any:
    """Prevent concurrent PyArrow conversions from separate browser sessions."""

    with DATAFRAME_SERIALIZATION_LOCK:
        return st.dataframe(data, **kwargs)


@st.cache_data(show_spinner="Discovering runs...")
def _cached_runs(path: str, signature: tuple[Any, ...]):
    del signature
    return discover_runs(path)


@st.cache_data(show_spinner="Loading evaluation sidecars...")
def _cached_evaluations(
    runs: dict[str, Any],
    evaluations_path: str,
    source_signature: tuple[Any, ...],
    evaluation_signature: tuple[Any, ...],
) -> pd.DataFrame:
    del source_signature, evaluation_signature
    return load_available_evaluations(runs, Path(evaluations_path))


@st.cache_data(show_spinner="Loading selected task records...")
def _cached_task_rows(
    run: Any, benchmark_name: str, source_signature: tuple[Any, ...]
) -> pd.DataFrame:
    del source_signature
    return task_rows(run, benchmark_name)


def _render_overview(
    run_table: pd.DataFrame, accuracy_table: pd.DataFrame, selected_runs: list[str]
) -> None:
    st.subheader("Overview")
    selected_manifest = run_table[run_table["run_id"].isin(selected_runs)]
    if accuracy_table.empty:
        st.info("No valid evaluation sidecars yet. Generate them from the sidebar.")
    else:
        selected_accuracy = accuracy_table[
            accuracy_table["run_id"].isin(selected_runs)
        ]
        matrix = selected_accuracy.pivot(
            index="run_id", columns="benchmark_name", values="accuracy"
        )
        st.caption("Accuracy; unresolved grades are excluded and listed below.")
        _safe_dataframe(matrix.style.format("{:.1%}", na_rep="—"), width="stretch")
        _safe_dataframe(
            selected_accuracy,
            width="stretch",
            hide_index=True,
            column_config={"accuracy": st.column_config.NumberColumn(format="percent")},
        )

    with st.expander("Run configuration", expanded=False):
        config_columns = [
            "run_id",
            "model",
            "prompt_version",
            "benchmark_name",
            "status",
            "temperature",
            "seed",
            "thinking",
            "max_tokens",
        ]
        _safe_dataframe(
            selected_manifest[config_columns],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Operational metrics")
    operational_columns = [
        "run_id",
        "benchmark_name",
        "tasks",
        "successful_calls",
        "failed_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "wall_time_seconds",
        "tasks_per_second",
    ]
    _safe_dataframe(
        selected_manifest[operational_columns],
        width="stretch",
        hide_index=True,
    )


def _render_comparisons(
    evaluations: pd.DataFrame,
    selected_runs: list[str],
    selected_benchmarks: list[str],
) -> None:
    st.subheader("Pairwise comparisons")
    if len(selected_runs) < 2:
        st.info("Select at least two runs.")
        return
    if evaluations.empty:
        st.info("Generate evaluation sidecars before comparing accuracy.")
        return
    with st.spinner("Computing paired bootstrap intervals..."):
        results = pairwise_results(
            evaluations,
            selected_runs,
            selected_benchmarks,
            n_resamples=10_000,
            seed=42,
        )
    tables = pairwise_display_tables(results)
    if not tables:
        st.info("There are no run pairs to compare.")
        return
    st.caption(
        "Results are accuracy percentages. Diff is the third column minus the "
        "second column in percentage points. P-values are exact paired "
        "discordance tests with Holm correction across the displayed comparisons."
    )
    for (left_run, right_run), table in tables.items():
        diff_column = f"Diff ({right_run} − {left_run})"

        def color_p_value(row: pd.Series) -> list[str]:
            styles = [""] * len(row)
            p_value = row["p-value"]
            difference = row[diff_column]
            p_index = table.columns.get_loc("p-value")
            styles[p_index] = p_value_style(p_value, difference)
            return styles

        st.markdown(f"**{left_run} → {right_run}**")
        styled = (
            table.style.format(
                {
                    left_run: "{:.2f}%",
                    right_run: "{:.2f}%",
                    diff_column: "{:+.2f} pp",
                    "p-value": "{:.4g}",
                },
                na_rep="—",
            )
            .apply(color_p_value, axis=1)
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [("font-weight", "700")],
                    }
                ]
            )
            .hide(axis="index")
        )
        _safe_dataframe(styled, width="stretch", hide_index=True)

    complete = [result for result in results if result.paired_tasks]
    if complete:
        selected_result = st.selectbox(
            "Comparison detail",
            complete,
            format_func=lambda result: (
                f"{result.left_run} vs {result.right_run} — {result.scope}"
            ),
        )
        win_columns = st.columns(3)
        win_columns[0].metric(
            f"{selected_result.left_run} wins", selected_result.left_wins
        )
        win_columns[1].metric("Ties", selected_result.ties)
        win_columns[2].metric(
            f"{selected_result.right_run} wins", selected_result.right_wins
        )


def _render_task_explorer(
    runs: dict[str, Any],
    evaluations: pd.DataFrame,
    selected_runs: list[str],
    selected_benchmarks: list[str],
    source_signature: tuple[Any, ...],
) -> None:
    st.subheader("Task explorer")
    run_id = st.selectbox("Run", selected_runs, key="task_run")
    available = [
        benchmark
        for benchmark in selected_benchmarks
        if benchmark in runs[run_id].benchmarks
    ]
    if not available:
        st.info("The selected run has no selected benchmarks.")
        return
    benchmark = st.selectbox("Benchmark", available, key="task_benchmark")
    tasks = _cached_task_rows(runs[run_id], benchmark, source_signature)
    grades = evaluations[
        (evaluations["run_id"] == run_id)
        & (evaluations["benchmark_name"] == benchmark)
    ]
    grade_columns = [
        "task_id",
        "status",
        "score",
        "reason",
        "extracted_prediction",
        "extracted_ground_truth",
    ]
    if grades.empty:
        grades = pd.DataFrame(columns=grade_columns)
    merged = tasks.merge(grades[grade_columns], on="task_id", how="left")
    summary_columns = [
        "task_id",
        "status",
        "score",
        "reason",
        "latency_seconds",
        "output_tokens",
        "error",
        "category",
        "subcategory",
    ]
    _safe_dataframe(
        merged[summary_columns],
        width="stretch",
        hide_index=True,
    )
    task_id = st.selectbox("Task detail", merged["task_id"].tolist())
    row = merged[merged["task_id"] == task_id].iloc[0]
    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.markdown("**Ground truth**")
        st.code(str(row["ground_truth"]))
        st.markdown("**Extracted ground truth**")
        st.code(str(row.get("extracted_ground_truth")))
    with detail_columns[1]:
        st.markdown("**Extracted prediction**")
        st.code(str(row.get("extracted_prediction")))
        st.markdown("**Grade**")
        st.json(
            {
                "status": row.get("status"),
                "score": row.get("score"),
                "reason": row.get("reason"),
            }
        )
    st.markdown("**Solution**")
    st.code(str(row["solution"]))
    with st.expander("Reasoning", expanded=False):
        st.code(str(row["reasoning"]))


def main() -> None:
    st.title("Benchmark run comparison")
    with st.sidebar:
        st.header("Artifacts")
        runs_path = Path(
            st.text_input("Runs directory", value=str(resolve_runs_dir()))
        ).expanduser().resolve()
        evaluations_path = Path(
            st.text_input(
                "Evaluations directory", value=str(resolve_evaluations_dir())
            )
        ).expanduser().resolve()

    source_signature = _directory_signature(
        runs_path, ("*/manifest.json", "*/*.jsonl")
    )
    try:
        runs = _cached_runs(str(runs_path), source_signature)
    except ArtifactError as exc:
        st.error(str(exc))
        st.stop()
    if not runs:
        st.warning(f"No run manifests found under {runs_path}")
        st.stop()

    evaluation_signature = _directory_signature(evaluations_path, ("*/*.jsonl",))
    evaluations = _cached_evaluations(
        runs,
        str(evaluations_path),
        source_signature,
        evaluation_signature,
    )
    all_benchmarks = sorted(
        {benchmark for run in runs.values() for benchmark in run.benchmarks}
    )

    with st.sidebar:
        selected_runs = st.multiselect("Runs", sorted(runs), default=sorted(runs))
        selected_benchmarks = st.multiselect(
            "Benchmarks", all_benchmarks, default=all_benchmarks
        )
        evaluated_pairs = (
            set(zip(evaluations["run_id"], evaluations["benchmark_name"]))
            if not evaluations.empty
            else set()
        )
        missing = [
            (run_id, benchmark)
            for run_id in selected_runs
            for benchmark in selected_benchmarks
            if benchmark in runs[run_id].benchmarks
            and (run_id, benchmark) not in evaluated_pairs
        ]
        if missing:
            st.warning(f"{len(missing)} selected run/benchmark sidecars are missing.")
            if st.button("Evaluate missing results", type="primary"):
                progress = st.progress(0.0)
                total_artifacts = len(missing)
                for artifact_index, (run_id, benchmark) in enumerate(missing):
                    st.write(f"Evaluating {run_id}/{benchmark}")

                    def update(completed: int, total: int) -> None:
                        fraction = (
                            artifact_index + (completed / total if total else 1.0)
                        ) / total_artifacts
                        progress.progress(min(fraction, 1.0))

                    evaluate_artifact(
                        runs[run_id].benchmarks[benchmark],
                        evaluations_path,
                        progress=update,
                    )
                progress.progress(1.0)
                st.cache_data.clear()
                st.rerun()

    if not selected_runs:
        st.info("Select at least one run.")
        st.stop()
    if not selected_benchmarks:
        st.info("Select at least one benchmark.")
        st.stop()

    tabs = st.tabs(["Overview", "Pairwise comparison", "Task explorer"])
    run_table = manifest_rows(runs)
    accuracy_table = accuracy_rows(evaluations)
    with tabs[0]:
        _render_overview(run_table, accuracy_table, selected_runs)
    with tabs[1]:
        _render_comparisons(evaluations, selected_runs, selected_benchmarks)
    with tabs[2]:
        _render_task_explorer(
            runs,
            evaluations,
            selected_runs,
            selected_benchmarks,
            source_signature,
        )


if __name__ == "__main__":
    main()
