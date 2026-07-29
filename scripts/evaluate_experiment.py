#!/usr/bin/env python3
"""Evaluate an existing run, optionally without updating comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Existing run directory name under results/runs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Override the directory containing run artifacts.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Evaluate the run without updating the comparison table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bootstrap_project(ROOT)
        experiment_parameters: dict[str, object] = {"run_id": args.run_id}
        if args.runs_dir is not None:
            experiment_parameters["runs_dir"] = str(args.runs_dir)
        runtime_parameters = {"experiment": experiment_parameters}
        with KedroSession.create(
            project_path=ROOT,
            extra_params=runtime_parameters,
        ) as session:
            session.run(
                pipeline_name=(
                    "evaluate" if args.score_only else "evaluate_compare"
                )
            )
    except Exception as exc:
        print(
            f"MATHAGENT_EVALUATION_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
