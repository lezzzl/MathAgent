"""Nodes for the run -> evaluate -> compare experiment workflow."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dashboard.artifacts import discover_runs
from dashboard.comparisons import add_new_runs_to_comparison_table

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _project_root(config: Mapping[str, Any]) -> Path:
    return _project_path(config.get("project_root", PROJECT_ROOT), PROJECT_ROOT)


def _runs_dir(config: Mapping[str, Any], project_root: Path) -> Path:
    return _project_path(config.get("runs_dir", "results/runs"), project_root)


def _comparison_path(config: Mapping[str, Any], project_root: Path) -> Path:
    return _project_path(
        config.get("comparisons_path", "results/comparisons.parquet"),
        project_root,
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read run manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Run manifest must contain a JSON object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generated_run_id(config: Mapping[str, Any]) -> str:
    agent_arguments = list(config.get("agent", {}).get("args", []))
    model = str(
        config.get("manifest", {}).get("model")
        or _argument_value(agent_arguments, "--model")
        or os.getenv("MODEL", "agent")
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-._") or "agent"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug}-{timestamp}"


def _validated_run_id(value: object) -> str:
    run_id = str(value)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    return run_id


def _command(
    script: str | Path,
    arguments: Sequence[object],
    project_root: Path,
) -> list[str]:
    script_path = _project_path(script, project_root)
    if not script_path.is_file():
        raise FileNotFoundError(f"Configured script does not exist: {script_path}")
    return [sys.executable, str(script_path), *(str(value) for value in arguments)]


def _argument_value(arguments: Sequence[object], option: str) -> str | None:
    """Return the final value passed for a conventional ``--option value`` pair."""

    result: str | None = None
    for index, argument in enumerate(arguments[:-1]):
        if str(argument) == option:
            result = str(arguments[index + 1])
    return result


def _record_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def run_agentic_loop(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run configured agent benchmark scripts and create a dashboard run manifest."""

    project_root = _project_root(config)
    runs_dir = _runs_dir(config, project_root)
    run_id = _validated_run_id(config.get("run_id") or _generated_run_id(config))
    run_directory = runs_dir / run_id
    manifest_path = run_directory / "manifest.json"
    agent_config = config.get("agent", {})
    benchmarks = agent_config.get("benchmarks", [])
    if not benchmarks:
        raise ValueError("experiment.agent.benchmarks must contain at least one item")
    if manifest_path.exists():
        raise FileExistsError(
            f"Run {run_id!r} already exists at {run_directory}; "
            "choose another RUN_ID or run only the evaluation stages"
        )

    common_args = list(agent_config.get("args", []))
    configured_manifest = dict(config.get("manifest", {}))
    configured_model = (
        _argument_value(common_args, "--model") or os.getenv("MODEL", "unknown")
    )
    configured_pipeline = (
        _argument_value(common_args, "--pipeline") or "default"
    )
    now = _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "model": configured_manifest.pop("model", configured_model),
        "pipeline": configured_manifest.pop("pipeline", configured_pipeline),
        **configured_manifest,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "benchmarks": {},
    }
    _write_json_atomic(manifest_path, manifest)

    try:
        for benchmark in benchmarks:
            name = str(benchmark["name"])
            output_name = str(benchmark.get("output") or f"{name.lower()}.jsonl")
            output_path = run_directory / Path(output_name).name
            arguments = [
                *common_args,
                *benchmark.get("args", []),
                "--output",
                str(output_path),
            ]
            command = _command(benchmark["script"], arguments, project_root)
            LOGGER.info("Running agent benchmark %s", name)
            subprocess.run(command, cwd=project_root, check=True)
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"Agent benchmark {name!r} did not create {output_path}"
                )
            manifest["benchmarks"][name] = {
                "output": str(output_path),
                "status": "completed",
                "tasks": _record_count(output_path),
                "updated_at": _utc_now(),
            }
            manifest["updated_at"] = _utc_now()
            _write_json_atomic(manifest_path, manifest)
    except BaseException:
        manifest["status"] = "failed"
        manifest["updated_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    manifest["updated_at"] = _utc_now()
    _write_json_atomic(manifest_path, manifest)
    LOGGER.info("Agent run %s completed: %s", run_id, run_directory)
    return {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "manifest_path": str(manifest_path),
    }


def _latest_run_directory(runs_dir: Path) -> Path:
    manifests = list(runs_dir.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No run manifests found under {runs_dir}")
    latest = max(manifests, key=lambda path: path.stat().st_mtime_ns)
    return latest.parent


def _selected_run(
    config: Mapping[str, Any],
    upstream: Mapping[str, Any] | None = None,
) -> Path:
    if upstream is not None:
        return Path(str(upstream["run_directory"])).resolve()
    project_root = _project_root(config)
    runs_dir = _runs_dir(config, project_root)
    configured_run_id = config.get("run_id")
    if configured_run_id:
        return runs_dir / _validated_run_id(configured_run_id)
    latest = _latest_run_directory(runs_dir)
    LOGGER.warning("RUN_ID is not set; selected latest run %s", latest.name)
    return latest


def _local_output(run_directory: Path, entry: Mapping[str, Any]) -> Path:
    output = entry.get("output")
    if not output:
        raise ValueError(f"Benchmark entry has no output path in {run_directory}")
    candidate = run_directory / Path(str(output)).name
    if not candidate.is_file():
        raise FileNotFoundError(f"Benchmark output does not exist: {candidate}")
    return candidate


def _jsonl_is_scored(path: Path) -> bool:
    records = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            if record.get("score") is None and record.get("is_correct") is None:
                return False
    return records > 0


def _evaluator_for(
    evaluation_config: Mapping[str, Any],
    benchmark_name: str,
) -> Mapping[str, Any]:
    overrides = evaluation_config.get("by_benchmark", {})
    evaluator = overrides.get(benchmark_name, evaluation_config.get("default"))
    if not evaluator:
        raise ValueError(f"No evaluator configured for benchmark {benchmark_name!r}")
    return evaluator


def _evaluate(config: Mapping[str, Any], run_directory: Path) -> dict[str, Any]:
    project_root = _project_root(config)
    manifest_path = run_directory / "manifest.json"
    manifest = _read_manifest(manifest_path)
    evaluation_config = config.get("evaluation", {})
    evaluated: list[str] = []
    skipped: list[str] = []

    manifest["evaluation"] = {
        "status": "running",
        "updated_at": _utc_now(),
    }
    _write_json_atomic(manifest_path, manifest)
    try:
        for name, entry in manifest.get("benchmarks", {}).items():
            input_path = _local_output(run_directory, entry)
            if _jsonl_is_scored(input_path):
                skipped.append(str(name))
                continue

            evaluator = _evaluator_for(evaluation_config, str(name))
            suffix = str(evaluator.get("output_suffix", "_verified"))
            output_path = input_path.with_name(f"{input_path.stem}{suffix}.jsonl")
            arguments = [
                str(input_path),
                "--output",
                str(output_path),
                *evaluation_config.get("args", []),
                *evaluator.get("args", []),
            ]
            command = _command(evaluator["script"], arguments, project_root)
            LOGGER.info("Evaluating benchmark %s", name)
            subprocess.run(command, cwd=project_root, check=True)
            if not output_path.is_file() or not _jsonl_is_scored(output_path):
                raise ValueError(
                    f"Evaluator for {name!r} did not produce scored JSONL: "
                    f"{output_path}"
                )
            entry.setdefault("raw_output", str(input_path))
            entry["output"] = str(output_path)
            entry["evaluation"] = {
                "status": "completed",
                "script": str(evaluator["script"]),
                "updated_at": _utc_now(),
            }
            evaluated.append(str(name))
            manifest["evaluation"]["updated_at"] = _utc_now()
            _write_json_atomic(manifest_path, manifest)
    except BaseException:
        manifest["evaluation"]["status"] = "failed"
        manifest["evaluation"]["updated_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest["evaluation"]["status"] = "completed"
    manifest["evaluation"]["updated_at"] = _utc_now()
    _write_json_atomic(manifest_path, manifest)
    LOGGER.info(
        "Evaluation completed for %s: evaluated=%d skipped=%d",
        manifest.get("run_id") or run_directory.name,
        len(evaluated),
        len(skipped),
    )
    return {
        "run_id": str(manifest.get("run_id") or run_directory.name),
        "run_directory": str(run_directory),
        "manifest_path": str(manifest_path),
        "evaluated_benchmarks": evaluated,
        "skipped_benchmarks": skipped,
    }


def evaluate_agent_run(
    agent_run: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the run produced by stage 1."""

    return _evaluate(config, _selected_run(config, agent_run))


def evaluate_selected_run(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a configured existing run, or the latest run as a fallback."""

    return _evaluate(config, _selected_run(config))


def update_comparison_table(
    config: Mapping[str, Any],
    evaluated_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add missing comparisons between all discovered pre-scored runs."""

    # The chained input provides ordering; filesystem discovery is authoritative.
    del evaluated_run
    project_root = _project_root(config)
    runs_dir = _runs_dir(config, project_root)
    output = _comparison_path(config, project_root)
    runs = discover_runs(runs_dir)
    comparison_config = config.get("comparison", {})
    update = add_new_runs_to_comparison_table(
        runs,
        output,
        n_resamples=comparison_config.get("n_resamples"),
        seed=int(comparison_config.get("seed", 42)),
    )
    LOGGER.info(
        "Comparison table updated: %s (newly represented runs=%d, added rows=%d)",
        output,
        len(update.added_runs),
        update.comparison_rows_added,
    )
    return {
        "comparison_path": str(output),
        "added_runs": list(update.added_runs),
        "skipped_runs": list(update.skipped_runs),
        "comparison_rows_added": update.comparison_rows_added,
        "comparison_rows_total": len(update.table),
    }
