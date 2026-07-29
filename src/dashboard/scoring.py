"""Score discovered run artifacts through the existing experiment evaluator."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dashboard.artifacts import PROJECT_ROOT, RunArtifact, load_task_records


@dataclass(frozen=True)
class ScoringFailure:
    """One run that the evaluator could not score."""

    run_id: str
    detail: str


@dataclass(frozen=True)
class ScoringUpdate:
    """Result of attempting to score every discovered run that needs it."""

    scored_runs: tuple[str, ...]
    skipped_runs: tuple[str, ...]
    failed_runs: tuple[ScoringFailure, ...]


def _run_needs_scoring(run: RunArtifact) -> bool:
    """Return whether any benchmark has a missing per-task score."""

    for artifact in run.benchmarks.values():
        records = load_task_records(artifact.path)
        if not records or any(
            record.get("score") is None and record.get("is_correct") is None
            for record in records
        ):
            return True
    return False


def _failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Extract a concise evaluator error for the Streamlit sidebar."""

    output = (result.stderr or result.stdout or "").strip()
    return output.splitlines()[-1] if output else f"exit code {result.returncode}"


def score_runs(
    runs: Mapping[str, RunArtifact],
    runs_dir: Path,
) -> ScoringUpdate:
    """Run the existing score-only evaluation pipeline for unscored runs."""

    scored: list[str] = []
    skipped: list[str] = []
    failed: list[ScoringFailure] = []
    script = PROJECT_ROOT / "scripts" / "evaluate_experiment.py"

    for run_id in sorted(runs):
        if not _run_needs_scoring(runs[run_id]):
            skipped.append(run_id)
            continue
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-id",
                    run_id,
                    "--runs-dir",
                    str(runs_dir),
                    "--score-only",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            failed.append(
                ScoringFailure(run_id, f"{type(exc).__name__}: {exc}")
            )
            continue
        if result.returncode == 0:
            scored.append(run_id)
        else:
            failed.append(ScoringFailure(run_id, _failure_detail(result)))

    return ScoringUpdate(tuple(scored), tuple(skipped), tuple(failed))
