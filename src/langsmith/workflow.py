"""LangSmith publishing and pairwise report orchestration."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import UUID, uuid5

from langsmith.evaluation import evaluate_comparative
from langsmith.utils import LangSmithNotFoundError

from artifact_io import LocalRun, TaskRecord, load_local_run
from bootstrap_stats import SCHEMA_VERSION, compare_score_maps
from grading import Grade, grade_answer


EXAMPLE_NAMESPACE = UUID("c8988611-58a6-4756-93a6-99ce3ddf4ac7")
DEFAULT_SOURCE_DATASET = "mathagent-benchmarks-v1"
DEFAULT_REPORT_DATASET = "mathagent-comparison-reports-v1"
SOURCE_KIND = "mathagent-source"
REPORT_KIND = "mathagent-comparison-report"


def _metadata(value: Any) -> dict[str, Any]:
    extra = getattr(value, "extra", None) or {}
    return dict(extra.get("metadata") or {})


def _source_project_name(run_id: str) -> str:
    return f"mathagent::{run_id}"


def _experiment_url(client: Any, project: Any) -> str:
    runs = list(
        client.list_runs(project_id=project.id, is_root=True, limit=1)
    )
    if runs:
        return client.get_run_url(run=runs[0], project_id=project.id)
    return f"LangSmith project {project.name} ({project.id})"


def _ensure_dataset(
    client: Any, name: str, description: str, *, metadata: dict[str, Any]
) -> Any:
    try:
        return client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        return client.create_dataset(name, description=description, metadata=metadata)


def _get_or_create_examples(
    client: Any, dataset_id: Any, definitions: list[dict[str, Any]]
) -> list[Any]:
    """Reuse canonical task examples and create only definitions not yet present."""

    example_ids = [definition["id"] for definition in definitions]
    wanted_ids = {str(example_id) for example_id in example_ids}
    existing = {
        str(example.id): example
        for example in client.list_examples(dataset_id=dataset_id)
        if str(example.id) in wanted_ids
    }
    missing: list[dict[str, Any]] = []
    for definition in definitions:
        example = existing.get(str(definition["id"]))
        if example is None:
            missing.append(definition)
            continue

        expected_inputs = definition.get("inputs") or {}
        actual_inputs = example.inputs or {}
        for field in ("example_key", "benchmark_name", "task_id", "report_name"):
            if (
                field in expected_inputs
                and actual_inputs.get(field) != expected_inputs[field]
            ):
                raise ValueError(
                    f"LangSmith example ID {definition['id']} has conflicting "
                    f"{field}"
                )

        expected_solution = (definition.get("outputs") or {}).get("solution")
        actual_solution = (example.outputs or {}).get("solution")
        if (
            expected_solution is not None
            and actual_solution is not None
            and expected_solution != actual_solution
        ):
            raise ValueError(
                f"LangSmith example {expected_inputs} has a conflicting "
                "reference solution"
            )

    if missing:
        client.create_examples(dataset_id=dataset_id, examples=missing)

    synchronized = {
        str(example.id): example
        for example in client.list_examples(dataset_id=dataset_id)
        if str(example.id) in wanted_ids
    }
    absent = [
        str(example_id)
        for example_id in example_ids
        if str(example_id) not in synchronized
    ]
    if absent:
        raise RuntimeError(
            f"LangSmith did not return synchronized examples: {', '.join(absent)}"
        )
    return [synchronized[str(example_id)] for example_id in example_ids]


def _example_id(task: TaskRecord) -> UUID:
    return uuid5(EXAMPLE_NAMESPACE, task.example_key)


def _grade_tasks(
    run: LocalRun, grader: Callable[[str | None, str | None], Grade]
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for task in run.tasks:
        grade = grader(task.solution, task.ground_truth)
        outputs[task.example_key] = {
            "benchmark_name": task.benchmark_name,
            "task_id": task.task_id,
            "solution": task.solution,
            "reasoning": task.reasoning,
            "grading_status": grade.status,
            "grading_reason": grade.reason,
            "correct": grade.correct,
        }
    return outputs


def publish_source_run(
    client: Any,
    run: LocalRun,
    *,
    dataset_name: str = DEFAULT_SOURCE_DATASET,
    grader: Callable[[str | None, str | None], Grade] = grade_answer,
) -> Any:
    """Publish a local artifact as an idempotently named LangSmith experiment."""

    dataset = _ensure_dataset(
        client,
        dataset_name,
        "Canonical MathAgent tasks from all benchmark suites.",
        metadata={"schema_version": SCHEMA_VERSION, "kind": SOURCE_KIND},
    )
    project_name = _source_project_name(run.run_id)
    existing = list(client.list_projects(name=project_name, limit=1))
    if existing and list(
        client.list_runs(project_id=existing[0].id, is_root=True, limit=1)
    ):
        return existing[0]

    examples = [
        {
            "id": _example_id(task),
            "inputs": {
                "example_key": task.example_key,
                "benchmark_name": task.benchmark_name,
                "task_id": task.task_id,
                "problem": task.problem,
            },
            "outputs": {"solution": task.ground_truth},
            "metadata": {
                "benchmark_name": task.benchmark_name,
                "task_id": task.task_id,
                "example_key": task.example_key,
            },
        }
        for task in run.tasks
    ]
    dataset_examples = _get_or_create_examples(client, dataset.id, examples)
    outputs = _grade_tasks(run, grader)

    def replay(inputs: dict[str, Any]) -> dict[str, Any]:
        return outputs[str(inputs["example_key"])]

    def accuracy(outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": "accuracy", "score": outputs.get("correct")}

    manifest_metadata = {
        key: run.manifest.get(key)
        for key in ("model", "pipeline", "prompt_version", "generation", "serving")
    }
    project = client.create_project(
        project_name,
        upsert=True,
        reference_dataset_id=dataset.id,
        metadata={
            "kind": SOURCE_KIND,
            "schema_version": SCHEMA_VERSION,
            "local_run_id": run.run_id,
            **manifest_metadata,
        },
    )
    client.evaluate(
        replay,
        data=dataset_examples,
        evaluators=[accuracy],
        experiment=project,
        metadata={"local_run_id": run.run_id, "schema_version": SCHEMA_VERSION},
        max_concurrency=0,
    )
    return client.read_project(project_id=project.id)


def compatible_source_projects(
    client: Any, *, dataset_id: Any, exclude_project_id: Any | None = None
) -> list[Any]:
    """Return compatible source experiments in deterministic order."""

    projects = client.list_projects(reference_dataset_id=dataset_id)
    compatible = []
    seen_run_ids: set[str] = set()
    for project in sorted(projects, key=lambda item: (item.start_time, item.name)):
        metadata = _metadata(project)
        run_id = metadata.get("local_run_id")
        if (
            project.id == exclude_project_id
            or metadata.get("kind") != SOURCE_KIND
            or metadata.get("schema_version") != SCHEMA_VERSION
            or not run_id
            or run_id in seen_run_ids
        ):
            continue
        seen_run_ids.add(str(run_id))
        compatible.append(project)
    return compatible


def _score_map(client: Any, project: Any) -> dict[tuple[str, str], bool | None]:
    scores: dict[tuple[str, str], bool | None] = {}
    runs = client.list_runs(project_id=project.id, is_root=True)
    for run in runs:
        output = run.outputs or {}
        benchmark = output.get("benchmark_name")
        task_id = output.get("task_id")
        if benchmark is None or task_id is None:
            continue
        value = output.get("correct")
        scores[(str(benchmark), str(task_id))] = (
            value if isinstance(value, bool) else None
        )
    return scores


def report_project_name(
    left: Any, right: Any, *, resamples: int, seed: int
) -> str:
    identity = json.dumps(
        {
            "left": str(left.id),
            "right": str(right.id),
            "resamples": resamples,
            "seed": seed,
            "schema": SCHEMA_VERSION,
        },
        sort_keys=True,
    )
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"mathagent-compare::{left.name}::{right.name}::{suffix}"


def _pairwise_accuracy(runs: list[Any]) -> dict[str, Any]:
    """Score a native LangSmith pair by resolved per-task correctness."""

    if len(runs) != 2:
        raise ValueError("pairwise accuracy requires exactly two runs")
    correctness = [
        (run.outputs or {}).get("correct")
        for run in runs
    ]
    if not all(isinstance(value, bool) for value in correctness):
        return {
            "key": "pairwise_accuracy",
            "scores": {},
            "comment": "Excluded because at least one correctness grade is unresolved.",
        }
    if correctness[0] == correctness[1]:
        scores = {runs[0].id: 0.5, runs[1].id: 0.5}
        comment = "Tie: both answers have the same correctness grade."
    elif correctness[0]:
        scores = {runs[0].id: 1.0, runs[1].id: 0.0}
        comment = "Run-1 is correct and Run-2 is incorrect."
    else:
        scores = {runs[0].id: 0.0, runs[1].id: 1.0}
        comment = "Run-2 is correct and Run-1 is incorrect."
    return {"key": "pairwise_accuracy", "scores": scores, "comment": comment}


def publish_native_comparison(
    client: Any,
    left: Any,
    right: Any,
) -> tuple[Any, str]:
    """Create LangSmith's native comparative experiment for one ordered pair."""

    prefix = f"mathagent-native::{left.name}::{right.name}"
    results = evaluate_comparative(
        (left.id, right.id),
        evaluators=[_pairwise_accuracy],
        experiment_prefix=prefix,
        description=(
            "Native paired correctness comparison. Run-1 is the existing "
            "experiment and Run-2 is the newer experiment."
        ),
        max_concurrency=5,
        client=client,
        metadata={
            "kind": "mathagent-native-comparison",
            "schema_version": SCHEMA_VERSION,
            "left_experiment_id": str(left.id),
            "right_experiment_id": str(right.id),
        },
        randomize_order=False,
    )
    return results.comparative_experiment, results.url


def _ensure_native_comparison(
    client: Any, report_project: Any, left: Any, right: Any
) -> Any:
    """Backfill and remember the native comparison associated with a report."""

    metadata = _metadata(report_project)
    if metadata.get("native_comparative_experiment_id"):
        return report_project
    comparative, url = publish_native_comparison(client, left, right)
    client.update_project(
        report_project.id,
        metadata={
            **metadata,
            "native_comparative_experiment_id": str(comparative.id),
            "native_comparative_experiment_name": comparative.name,
            "native_comparative_url": url,
        },
    )
    return client.read_project(project_id=report_project.id)


def publish_comparison(
    client: Any,
    left: Any,
    right: Any,
    *,
    report_dataset_name: str = DEFAULT_REPORT_DATASET,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[Any, str]:
    """Create or reuse the deterministic report for one ordered run pair."""

    name = report_project_name(left, right, resamples=resamples, seed=seed)
    existing = list(client.list_projects(name=name, limit=1))
    if existing and list(
        client.list_runs(project_id=existing[0].id, is_root=True, limit=1)
    ):
        project = _ensure_native_comparison(
            client, existing[0], left, right
        )
        return project, _experiment_url(client, project)

    rows = compare_score_maps(
        _score_map(client, left),
        _score_map(client, right),
        resamples=resamples,
        seed=seed,
    )
    left_metadata, right_metadata = _metadata(left), _metadata(right)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_1": {
            "experiment_id": str(left.id),
            "experiment_name": left.name,
            "local_run_id": left_metadata.get("local_run_id"),
        },
        "run_2": {
            "experiment_id": str(right.id),
            "experiment_name": right.name,
            "local_run_id": right_metadata.get("local_run_id"),
        },
        "statistics": {"method": "paired_centered_null_bootstrap", "resamples": resamples, "seed": seed},
        "benchmarks": [row.to_dict() for row in rows],
    }
    dataset = _ensure_dataset(
        client,
        report_dataset_name,
        "Materialized pairwise benchmark comparison reports.",
        metadata={"schema_version": SCHEMA_VERSION, "kind": REPORT_KIND},
    )
    sentinel_id = uuid5(EXAMPLE_NAMESPACE, f"report:{name}")
    example = _get_or_create_examples(
        client,
        dataset.id,
        [
            {
                "id": sentinel_id,
                "inputs": {"report_name": name},
                "outputs": {},
                "metadata": {"kind": REPORT_KIND},
            }
        ],
    )[0]
    project = client.create_project(
        name,
        upsert=True,
        reference_dataset_id=dataset.id,
        metadata={
            "kind": REPORT_KIND,
            "schema_version": SCHEMA_VERSION,
            "left_experiment_id": str(left.id),
            "right_experiment_id": str(right.id),
            "resamples": resamples,
            "seed": seed,
        },
    )
    client.evaluate(
        lambda _inputs: payload,
        data=[example],
        experiment=project,
        metadata={"kind": REPORT_KIND, "schema_version": SCHEMA_VERSION},
        max_concurrency=0,
    )
    project = client.read_project(project_id=project.id)
    project = _ensure_native_comparison(client, project, left, right)
    return project, _experiment_url(client, project)


def publish_and_compare(
    client: Any,
    *,
    results_dir: Path,
    run_id: str,
    source_dataset_name: str = DEFAULT_SOURCE_DATASET,
    report_dataset_name: str = DEFAULT_REPORT_DATASET,
    resamples: int = 10_000,
    seed: int = 42,
    grader: Callable[[str | None, str | None], Grade] = grade_answer,
) -> tuple[Any, list[tuple[Any, str]], list[tuple[str, Exception]]]:
    """Publish a run and compare it independently with every existing run."""

    run = load_local_run(results_dir, run_id)
    project = publish_source_run(
        client, run, dataset_name=source_dataset_name, grader=grader
    )
    reports: list[tuple[Any, str]] = []
    failures: list[tuple[str, Exception]] = []
    for previous in compatible_source_projects(
        client,
        dataset_id=project.reference_dataset_id,
        exclude_project_id=project.id,
    ):
        try:
            reports.append(
                publish_comparison(
                    client,
                    previous,
                    project,
                    report_dataset_name=report_dataset_name,
                    resamples=resamples,
                    seed=seed,
                )
            )
        except Exception as exc:
            failures.append((previous.name, exc))
    return project, reports, failures


def resolve_source_project(client: Any, identifier: str) -> Any:
    """Resolve an experiment name or UUID and enforce source compatibility."""

    try:
        project_id = UUID(identifier)
    except ValueError:
        project_id = None
    try:
        project = (
            client.read_project(project_id=project_id)
            if project_id is not None
            else client.read_project(project_name=identifier)
        )
    except LangSmithNotFoundError as exc:
        identifier_kind = "ID" if project_id is not None else "name"
        raise ValueError(
            f"Source experiment with {identifier_kind} {identifier!r} was not found"
        ) from exc
    metadata = _metadata(project)
    if (
        metadata.get("kind") != SOURCE_KIND
        or metadata.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(
            f"{identifier!r} is not a compatible MathAgent source experiment"
        )
    return project
