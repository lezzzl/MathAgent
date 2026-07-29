"""Streamlit application for comparing pre-scored benchmark runs."""

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
    COMPARISONS_PATH_ENV,
    RUNS_DIR_ENV,
    discover_runs,
)
from dashboard.comparisons import (
    add_new_runs_to_comparison_table,
    load_comparison_table,
)
from dashboard.data import (
    baseline_comparison_table,
    comparison_run_ids,
    p_value_style,
    two_run_comparison_table,
)
from dashboard.runtime import (
    DATAFRAME_SERIALIZATION_LOCK,
    use_system_arrow_memory_pool,
)


use_system_arrow_memory_pool()
st.set_page_config(page_title="MathAgent run comparison", layout="wide")

STREAMLIT_DEFAULT_RUNS_DIR = Path("/mnt/storage-1/MathAgent/results/runs")
STREAMLIT_DEFAULT_COMPARISONS_PATH = Path(
    "/mnt/storage-1/MathAgent/results/comparison/table.parquet"
)


def _streamlit_path_default(environment_variable: str, fallback: Path) -> str:
    """Return an environment override or the Streamlit-specific path default."""

    return os.getenv(environment_variable, str(fallback))


def _directory_signature(path: Path, patterns: tuple[str, ...]) -> tuple[Any, ...]:
    """Build a cheap cache key that changes when an artifact is rewritten."""

    if not path.exists():
        return (str(path), "missing")
    signature: list[Any] = [str(path)]
    for pattern in patterns:
        for item in sorted(path.glob(pattern)):
            stat = item.stat()
            signature.append(
                (str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns)
            )
    return tuple(signature)


def _safe_dataframe(data: Any, **kwargs: Any) -> Any:
    """Prevent concurrent PyArrow conversions from separate browser sessions."""

    with DATAFRAME_SERIALIZATION_LOCK:
        return st.dataframe(data, **kwargs)


@st.cache_data(show_spinner="Discovering runs...")
def _cached_runs(path: str, signature: tuple[Any, ...]):
    del signature
    return discover_runs(path)


def _render_two_run_view(comparisons: pd.DataFrame, run_ids: list[str]) -> None:
    st.subheader("Side-by-side comparison")
    selectors = st.columns(2)
    with selectors[0]:
        left_run = st.selectbox("Left run", run_ids, key="left_run")
    right_options = [run_id for run_id in run_ids if run_id != left_run]
    with selectors[1]:
        right_run = st.selectbox("Right run", right_options, key="right_run")

    table = two_run_comparison_table(comparisons, left_run, right_run)
    if table.empty:
        st.info("These runs have no benchmark comparisons.")
        return

    def color_p_value(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        p_index = table.columns.get_loc("p-value")
        styles[p_index] = p_value_style(row["p-value"], row["Diff"])
        return styles

    styled = (
        table.style.format(
            {
                left_run: "{:.2f}%",
                right_run: "{:.2f}%",
                "Diff": "{:+.2f} pp",
                "p-value": "{:.4g}",
            },
            na_rep="—",
        )
        .apply(color_p_value, axis=1)
        .hide(axis="index")
    )
    st.caption(
        "Diff is right minus left. P-values are paired exact sign tests with "
        "Holm correction across run pairs within each benchmark."
    )
    _safe_dataframe(styled, width="stretch", hide_index=True)


def _render_baseline_view(comparisons: pd.DataFrame, run_ids: list[str]) -> None:
    st.subheader("Compare runs against a baseline")
    baseline_run = st.selectbox("Baseline run", run_ids, key="baseline_run")
    available = [run_id for run_id in run_ids if run_id != baseline_run]
    default = available[: min(3, len(available))]
    compared_runs = st.multiselect(
        "Runs to compare",
        available,
        default=default,
        key="compared_runs",
    )
    if not compared_runs:
        st.info("Select at least one run to compare.")
        return

    table, p_values = baseline_comparison_table(
        comparisons, baseline_run, compared_runs
    )
    if table.empty:
        st.info("The selected runs have no benchmark comparisons.")
        return

    p_value_index = p_values.set_index("Benchmark")

    def color_differences(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        benchmark = row["Benchmark"]
        for run_id in compared_runs:
            column_index = table.columns.get_loc(run_id)
            p_value = (
                p_value_index.at[benchmark, run_id]
                if benchmark in p_value_index.index
                else None
            )
            styles[column_index] = p_value_style(p_value, row[run_id])
        return styles

    formats = {
        baseline_run: "{:.2f}%",
        **{run_id: "{:+.2f} pp" for run_id in compared_runs},
    }
    styled = (
        table.style.format(formats, na_rep="—")
        .apply(color_differences, axis=1)
        .hide(axis="index")
    )
    st.caption(
        "The baseline column shows its average score. Every other column is that "
        "run's difference from the baseline; significant improvements are green "
        "and significant regressions are red."
    )
    _safe_dataframe(styled, width="stretch", hide_index=True)


@st.fragment(run_every="2s")
def _render_comparison_views(comparisons_path: str) -> None:
    """Poll the comparison table so external updates appear automatically."""

    path = Path(comparisons_path)
    try:
        comparisons = load_comparison_table(path)
    except ArtifactError as exc:
        st.error(str(exc))
        return
    run_ids = comparison_run_ids(comparisons)
    if len(run_ids) < 2:
        st.info(
            "The comparison table does not contain a run pair yet. Add a "
            "pre-scored run from the sidebar."
        )
        return

    first_view, second_view = st.tabs(
        ["Two-run comparison", "Baseline comparison"]
    )
    with first_view:
        _render_two_run_view(comparisons, run_ids)
    with second_view:
        _render_baseline_view(comparisons, run_ids)


def main() -> None:
    st.title("Benchmark run comparison")
    with st.sidebar:
        st.header("Comparison data")
        runs_path = Path(
            st.text_input(
                "Runs directory",
                value=_streamlit_path_default(
                    RUNS_DIR_ENV, STREAMLIT_DEFAULT_RUNS_DIR
                ),
            )
        ).expanduser().resolve()
        update_comparisons = st.button(
            "Update comparison table", type="primary"
        )
        comparisons_path = Path(
            st.text_input(
                "Comparison table",
                value=_streamlit_path_default(
                    COMPARISONS_PATH_ENV,
                    STREAMLIT_DEFAULT_COMPARISONS_PATH,
                ),
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

    if update_comparisons:
        table_existed = comparisons_path.is_file()
        try:
            with st.spinner("Completing pairwise comparisons..."):
                update = add_new_runs_to_comparison_table(runs, comparisons_path)
        except ArtifactError as exc:
            st.sidebar.error(str(exc))
        else:
            if update.comparison_rows_added:
                action = "Updated" if table_existed else "Created"
                st.sidebar.success(
                    f"{action} comparison table with "
                    f"{update.comparison_rows_added} missing comparison rows."
                )
            elif not table_existed:
                st.sidebar.info(
                    "Created an empty comparison table; no compatible scored "
                    "run pairs were found."
                )
            else:
                st.sidebar.info(
                    "The comparison table already contains all compatible "
                    "scored run pairs."
                )
            if update.skipped_runs:
                st.sidebar.warning(
                    "Skipped unscored runs: " + ", ".join(update.skipped_runs)
                )

    with st.sidebar:
        st.caption(
            "Run records must already contain `score` or `is_correct`. "
            "The dashboard never grades solutions."
        )

    _render_comparison_views(str(comparisons_path))


if __name__ == "__main__":
    main()
