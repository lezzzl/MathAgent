"""Independent readers for MathAgent benchmark artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """Raised when a local benchmark run is incomplete or malformed."""


@dataclass(frozen=True)
class TaskRecord:
    benchmark_name: str
    task_id: str
    problem: str | None
    ground_truth: str | None
    solution: str | None
    reasoning: Any
    metadata: dict[str, Any]

    @property
    def example_key(self) -> str:
        return f"{self.benchmark_name}:{self.task_id}"


@dataclass(frozen=True)
class LocalRun:
    run_id: str
    manifest: dict[str, Any]
    tasks: tuple[TaskRecord, ...]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected an object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"Cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ArtifactError(f"Expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def load_local_run(results_dir: Path, run_id: str) -> LocalRun:
    """Load one run without importing the benchmark runner or dashboard."""

    run_dir = results_dir / run_id
    manifest = _read_object(run_dir / "manifest.json")
    manifest_run_id = str(manifest.get("run_id") or run_dir.name)
    if manifest_run_id != run_id:
        raise ArtifactError(
            f"Manifest run_id {manifest_run_id!r} does not match {run_id!r}"
        )
    benchmark_entries = manifest.get("benchmarks")
    if not isinstance(benchmark_entries, dict):
        raise ArtifactError("Manifest has no benchmark mapping")

    tasks: list[TaskRecord] = []
    seen: set[str] = set()
    for benchmark_name, entry in sorted(benchmark_entries.items()):
        if not isinstance(entry, dict):
            raise ArtifactError(f"Invalid benchmark entry {benchmark_name!r}")
        output = entry.get("output")
        candidates = sorted(run_dir.glob("*.jsonl"))
        path = (
            run_dir / Path(str(output)).name
            if output and (run_dir / Path(str(output)).name).is_file()
            else None
        )
        if path is None:
            normalized = re.sub(r"[^a-z0-9]+", "", str(benchmark_name).lower())
            path = next(
                (
                    candidate
                    for candidate in candidates
                    if re.sub(r"[^a-z0-9]+", "", candidate.stem.lower())
                    == normalized
                ),
                None,
            )
        if path is None:
            for candidate in candidates:
                candidate_rows = _read_jsonl(candidate)
                if any(
                    str(row.get("benchmark_name")) == str(benchmark_name)
                    for row in candidate_rows[:1]
                ):
                    path = candidate
                    break
        if path is None:
            raise ArtifactError(f"No JSONL output for benchmark {benchmark_name!r}")
        rows = _read_jsonl(path)
        for row in rows:
            actual_benchmark = str(row.get("benchmark_name") or benchmark_name)
            if actual_benchmark != str(benchmark_name):
                continue
            if "task_id" not in row:
                raise ArtifactError(f"Missing task_id in {path}")
            task = TaskRecord(
                benchmark_name=actual_benchmark,
                task_id=str(row["task_id"]),
                problem=(
                    None if row.get("problem") is None else str(row.get("problem"))
                ),
                ground_truth=(
                    None
                    if row.get("ground_truth") is None
                    else str(row.get("ground_truth"))
                ),
                solution=(
                    None if row.get("solution") is None else str(row.get("solution"))
                ),
                reasoning=row.get("reasoning"),
                metadata=dict(row.get("metadata") or {}),
            )
            if task.example_key in seen:
                raise ArtifactError(f"Duplicate task {task.example_key!r}")
            seen.add(task.example_key)
            tasks.append(task)
    return LocalRun(run_id=run_id, manifest=manifest, tasks=tuple(tasks))
