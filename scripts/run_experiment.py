#!/usr/bin/env python3
"""Run the complete agent -> evaluate -> compare Kedro workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML experiment configuration. Its root must contain run_id.",
    )
    return parser


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load a root-level experiment mapping with a required run ID."""

    resolved = path.expanduser().resolve()
    try:
        with resolved.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read experiment config {resolved}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML in experiment config {resolved}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Experiment config must contain a YAML mapping at its root: {resolved}"
        )
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(
            f"Experiment config must contain a non-empty string run_id: {resolved}"
        )
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_config = load_experiment_config(args.config)
    bootstrap_project(ROOT)
    with KedroSession.create(
        project_path=ROOT,
        extra_params={"experiment": experiment_config},
    ) as session:
        session.run(pipeline_name="experiment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
