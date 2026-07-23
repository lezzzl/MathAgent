#!/usr/bin/env python3
"""Publish MathAgent runs to LangSmith and create pairwise reports."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langsmith import Client

from workflow import (
    DEFAULT_REPORT_DATASET,
    DEFAULT_SOURCE_DATASET,
    publish_and_compare,
    publish_comparison,
    resolve_source_project,
)


ROOT = Path(__file__).resolve().parents[2]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--report-dataset", default=DEFAULT_REPORT_DATASET)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", default=os.getenv("LANGSMITH_ENDPOINT"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--results-dir", type=Path, default=ROOT / "results" / "runs")
    _common(publish)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    _common(compare)
    return parser.parse_args(argv)


def _client(endpoint: str | None) -> Client:
    kwargs = {"api_url": endpoint} if endpoint else {}
    return Client(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resamples < 1:
        raise SystemExit("--resamples must be positive")
    client = _client(args.endpoint)
    if args.command == "compare":
        try:
            left = resolve_source_project(client, args.left)
            right = resolve_source_project(client, args.right)
        except ValueError as exc:
            print(f"Cannot compare experiments: {exc}", file=sys.stderr)
            return 2
        _project, url = publish_comparison(
            client,
            left,
            right,
            report_dataset_name=args.report_dataset,
            resamples=args.resamples,
            seed=args.seed,
        )
        print(url)
        return 0

    project, reports, failures = publish_and_compare(
        client,
        results_dir=args.results_dir.resolve(),
        run_id=args.run_id,
        source_dataset_name=args.source_dataset,
        report_dataset_name=args.report_dataset,
        resamples=args.resamples,
        seed=args.seed,
    )
    print(f"Published {project.name}")
    if not reports and not failures:
        print("No compatible existing runs; this run is the initial baseline.")
    for report, url in reports:
        print(f"Created {report.name}: {url}")
    for name, exc in failures:
        print(f"Comparison with {name} failed: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
