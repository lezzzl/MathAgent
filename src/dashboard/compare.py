"""Incrementally add pre-scored runs to the dashboard comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

from dashboard.artifacts import (
    ArtifactError,
    discover_runs,
    resolve_comparisons_path,
    resolve_runs_dir,
)
from dashboard.comparisons import add_new_runs_to_comparison_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing run folders (default: results/runs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Comparison table path (default: results/comparisons.parquet)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_dir = resolve_runs_dir(args.runs_dir)
    output = resolve_comparisons_path(args.output)
    try:
        runs = discover_runs(runs_dir)
        update = add_new_runs_to_comparison_table(runs, output)
    except ArtifactError as exc:
        print(f"error: {exc}")
        return 2

    if update.added_runs:
        print(f"new scored runs: {', '.join(update.added_runs)}")
    else:
        print("no new scored runs")
    if update.skipped_runs:
        print(
            "skipped runs without precomputed scores: "
            f"{', '.join(update.skipped_runs)}"
        )
    print(f"added comparison rows: {update.comparison_rows_added}")
    print(f"comparison table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
