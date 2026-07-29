from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dashboard.artifacts import PROJECT_ROOT, RunArtifact, load_run
from dashboard.scoring import score_runs


def _write_run(
    root: Path, run_id: str, scores: list[bool | None]
) -> RunArtifact:
    directory = root / run_id
    directory.mkdir(parents=True)
    output = directory / "bench.jsonl"
    output.write_text(
        "".join(
            json.dumps(
                {
                    "run_id": run_id,
                    "benchmark_name": "Bench",
                    "task_id": str(index),
                    "solution": "answer",
                    "is_correct": score,
                }
            )
            + "\n"
            for index, score in enumerate(scores)
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "benchmarks": {"Bench": {"output": str(output)}},
            }
        ),
        encoding="utf-8",
    )
    return load_run(directory)


def test_score_runs_evaluates_only_runs_with_missing_scores(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    scored = _write_run(runs_dir, "run-a", [True, False])
    unscored = _write_run(runs_dir, "run-b", [None, None])
    commands: list[tuple[list[str], Path]] = []

    def fake_run(command, *, cwd, **kwargs):
        commands.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("dashboard.scoring.subprocess.run", fake_run)

    update = score_runs(
        {scored.run_id: scored, unscored.run_id: unscored}, runs_dir
    )

    assert update.scored_runs == ("run-b",)
    assert update.skipped_runs == ("run-a",)
    assert update.failed_runs == ()
    assert commands == [
        (
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_experiment.py"),
                "--run-id",
                "run-b",
                "--runs-dir",
                str(runs_dir),
                "--score-only",
            ],
            PROJECT_ROOT,
        )
    ]


def test_score_runs_reports_evaluator_failure(tmp_path: Path, monkeypatch) -> None:
    runs_dir = tmp_path / "runs"
    run = _write_run(runs_dir, "run-a", [None])

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, "", "details\nlast error")

    monkeypatch.setattr("dashboard.scoring.subprocess.run", fake_run)

    update = score_runs({run.run_id: run}, runs_dir)

    assert update.scored_runs == ()
    assert update.skipped_runs == ()
    assert len(update.failed_runs) == 1
    assert update.failed_runs[0].run_id == "run-a"
    assert update.failed_runs[0].detail == "last error"


def test_score_runs_reports_process_start_failure(
    tmp_path: Path, monkeypatch
) -> None:
    runs_dir = tmp_path / "runs"
    run = _write_run(runs_dir, "run-a", [None])

    def fake_run(command, **kwargs):
        raise OSError("cannot start evaluator")

    monkeypatch.setattr("dashboard.scoring.subprocess.run", fake_run)

    update = score_runs({run.run_id: run}, runs_dir)

    assert update.scored_runs == ()
    assert update.failed_runs[0].detail == "OSError: cannot start evaluator"
