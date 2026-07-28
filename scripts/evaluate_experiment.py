#!/usr/bin/env python3
"""Evaluate an existing run and update the comparison table."""

from __future__ import annotations

import argparse
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap_project(ROOT)
    runtime_parameters = {"experiment": {"run_id": args.run_id}}
    with KedroSession.create(
        project_path=ROOT,
        extra_params=runtime_parameters,
    ) as session:
        session.run(pipeline_name="evaluate_compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
