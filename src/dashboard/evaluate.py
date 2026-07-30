"""CLI for creating mathematical correctness sidecars for benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from dashboard.artifacts import (
    ArtifactError,
    discover_runs,
    resolve_evaluations_dir,
    resolve_runs_dir,
)
from dashboard.evaluation import evaluate_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="Directory containing run folders")
    parser.add_argument("--evaluations-dir", type=Path, help="Sidecar cache directory")
    parser.add_argument("--run-id", action="append", help="Run to evaluate; repeatable")
    parser.add_argument(
        "--benchmark", action="append", help="Benchmark to evaluate; repeatable"
    )
    parser.add_argument("--force", action="store_true", help="Rebuild valid caches")
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=15.0,
        help="Hard grading timeout per task in seconds (default: 15)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_dir = resolve_runs_dir(args.results_dir)
    evaluations_dir = resolve_evaluations_dir(args.evaluations_dir)
    try:
        runs = discover_runs(runs_dir)
    except ArtifactError as exc:
        print(f"error: {exc}")
        return 2

    selected_runs = set(args.run_id or runs)
    unknown_runs = selected_runs - set(runs)
    if unknown_runs:
        print(f"error: unknown run IDs: {', '.join(sorted(unknown_runs))}")
        return 2

    benchmark_filter = set(args.benchmark or ())
    evaluated_any = False
    for run_id in sorted(selected_runs):
        run = runs[run_id]
        for benchmark_name, artifact in sorted(run.benchmarks.items()):
            if benchmark_filter and benchmark_name not in benchmark_filter:
                continue
            evaluated_any = True
            print(f"evaluating {run_id}/{benchmark_name} ...", flush=True)

            def show_progress(completed: int, total: int) -> None:
                if completed == total or completed % 10 == 0:
                    print(f"  progress {completed}/{total}", flush=True)

            try:
                path, records, reused = evaluate_artifact(
                    artifact,
                    evaluations_dir,
                    force=args.force,
                    progress=show_progress,
                    timeout_seconds=args.task_timeout,
                )
            except KeyboardInterrupt:
                print("\ninterrupted; completed tasks were checkpointed")
                return 130
            except ArtifactError as exc:
                print(f"error: {exc}")
                return 2
            counts = {
                status: sum(record.status == status for record in records)
                for status in ("correct", "incorrect", "unresolved")
            }
            action = "reused" if reused else "wrote"
            print(
                f"{action} {path} "
                f"(correct={counts['correct']}, incorrect={counts['incorrect']}, "
                f"unresolved={counts['unresolved']})"
            )
    if not evaluated_any:
        print("error: no matching benchmark artifacts")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
