"""Read benchmark manifests, JSONL records, and evaluation sidecars."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEFAULT_EVALUATIONS_DIR = PROJECT_ROOT / "results" / "evaluations"
RUNS_DIR_ENV = "MATHAGENT_RESULTS_DIR"
EVALUATIONS_DIR_ENV = "MATHAGENT_EVALUATIONS_DIR"


class ArtifactError(ValueError):
    """Raised when a run artifact is malformed or internally inconsistent."""


@dataclass(frozen=True)
class BenchmarkArtifact:
    """A benchmark output belonging to a discovered run."""

    run_id: str
    benchmark_name: str
    path: Path
    manifest_entry: dict[str, Any]


@dataclass(frozen=True)
class RunArtifact:
    """A validated run manifest and its benchmark output files."""

    run_id: str
    directory: Path
    manifest: dict[str, Any]
    benchmarks: dict[str, BenchmarkArtifact]


def resolve_runs_dir(value: str | Path | None = None) -> Path:
    """Resolve an explicit, environment, or repository-default runs directory."""

    raw = value if value is not None else os.getenv(RUNS_DIR_ENV)
    return Path(raw).expanduser().resolve() if raw else DEFAULT_RUNS_DIR


def resolve_evaluations_dir(value: str | Path | None = None) -> Path:
    """Resolve an explicit, environment, or repository-default cache directory."""

    raw = value if value is not None else os.getenv(EVALUATIONS_DIR_ENV)
    return Path(raw).expanduser().resolve() if raw else DEFAULT_EVALUATIONS_DIR


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object with a useful artifact-specific error."""

    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object in {path}")
    return value


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file and reject incomplete records."""

    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"Cannot open JSONL artifact {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"Invalid JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ArtifactError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            yield value


def load_task_records(path: Path) -> list[dict[str, Any]]:
    """Load one record per task and validate the comparison identity fields."""

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in iter_jsonl(path):
        try:
            key = (str(record["benchmark_name"]), str(record["task_id"]))
        except KeyError as exc:
            raise ArtifactError(f"Missing {exc.args[0]!r} in {path}") from exc
        if key in seen:
            raise ArtifactError(
                f"Duplicate task benchmark={key[0]!r} task_id={key[1]!r} in {path}"
            )
        seen.add(key)
        records.append(record)
    return records


def _benchmark_file(
    run_directory: Path,
    benchmark_name: str,
    entry: dict[str, Any],
) -> Path:
    """Find a benchmark output without trusting machine-specific manifest paths."""

    output = entry.get("output")
    if output:
        local_candidate = run_directory / Path(str(output)).name
        if local_candidate.is_file():
            return local_candidate

    normalized = re.sub(r"[^a-z0-9]+", "", benchmark_name.lower())
    candidates = sorted(run_directory.glob("*.jsonl"))
    for candidate in candidates:
        candidate_name = re.sub(r"[^a-z0-9]+", "", candidate.stem.lower())
        if candidate_name == normalized:
            return candidate

    # The output name may differ from the display name. Inspect only the first row.
    for candidate in candidates:
        first = next(iter_jsonl(candidate), None)
        if first and str(first.get("benchmark_name")) == benchmark_name:
            return candidate
    raise ArtifactError(
        f"No JSONL output for benchmark {benchmark_name!r} in {run_directory}"
    )


def load_run(run_directory: Path) -> RunArtifact:
    """Load and validate a single run directory."""

    manifest_path = run_directory / "manifest.json"
    manifest = read_json(manifest_path)
    run_id = str(manifest.get("run_id") or run_directory.name)
    if run_id != run_directory.name:
        raise ArtifactError(
            f"Run ID {run_id!r} does not match directory {run_directory.name!r}"
        )
    manifest_benchmarks = manifest.get("benchmarks")
    if not isinstance(manifest_benchmarks, dict):
        raise ArtifactError(f"Manifest has no benchmark mapping: {manifest_path}")

    benchmarks: dict[str, BenchmarkArtifact] = {}
    for name, raw_entry in manifest_benchmarks.items():
        if not isinstance(raw_entry, dict):
            raise ArtifactError(f"Invalid benchmark entry {name!r}: {manifest_path}")
        path = _benchmark_file(run_directory, str(name), raw_entry)
        benchmarks[str(name)] = BenchmarkArtifact(
            run_id=run_id,
            benchmark_name=str(name),
            path=path,
            manifest_entry=raw_entry,
        )
    return RunArtifact(run_id, run_directory, manifest, benchmarks)


def discover_runs(runs_dir: str | Path | None = None) -> dict[str, RunArtifact]:
    """Discover run directories in deterministic order."""

    root = resolve_runs_dir(runs_dir)
    if not root.is_dir():
        raise ArtifactError(f"Runs directory does not exist: {root}")
    runs: dict[str, RunArtifact] = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        run = load_run(manifest_path.parent)
        if run.run_id in runs:
            raise ArtifactError(f"Duplicate run ID {run.run_id!r}")
        runs[run.run_id] = run
    return runs


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a content fingerprint for reliable cache invalidation."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_cache_path(
    evaluations_dir: Path,
    artifact: BenchmarkArtifact,
    source_fingerprint: str,
    grader_version: str,
) -> Path:
    """Build a cache filename that identifies both source and grading logic."""

    grader_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", grader_version)
    return (
        evaluations_dir
        / artifact.run_id
        / f"{artifact.path.stem}.{source_fingerprint[:16]}.{grader_slug}.jsonl"
    )


def find_evaluation_cache(
    evaluations_dir: Path,
    artifact: BenchmarkArtifact,
    source_fingerprint: str,
    grader_version: str,
) -> Path | None:
    """Return the exact valid cache for an artifact, if it exists."""

    candidate = evaluation_cache_path(
        evaluations_dir, artifact, source_fingerprint, grader_version
    )
    return candidate if candidate.is_file() else None
