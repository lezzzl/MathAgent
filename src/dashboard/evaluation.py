"""Mathematical answer grading and content-addressed evaluation caches."""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from dashboard.artifacts import (
    ArtifactError,
    BenchmarkArtifact,
    evaluation_cache_path,
    file_sha256,
    iter_jsonl,
    load_task_records,
)


EVALUATION_SCHEMA_VERSION = 1
GRADING_LOGIC_VERSION = "dashboard-v2"
DEFAULT_TASK_TIMEOUT_SECONDS = 15.0
DEFAULT_CHECKPOINT_EVERY = 10


@dataclass(frozen=True)
class EvaluationRecord:
    """A compact, auditable grade for one benchmark task."""

    schema_version: int
    source_fingerprint: str
    grader_version: str
    run_id: str
    benchmark_name: str
    task_id: str
    status: str
    score: bool | None
    extracted_prediction: str | None
    extracted_ground_truth: str | None
    reason: str | None


def grader_version() -> str:
    """Return a stable cache key for the dependency and local grading policy."""

    try:
        package_version = version("math-verify")
    except PackageNotFoundError:
        package_version = "unavailable"
    return f"math-verify-{package_version}.{GRADING_LOGIC_VERSION}"


def _short_repr(value: Any, limit: int = 1000) -> str | None:
    if value is None:
        return None
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def grade_answer(
    *,
    run_id: str,
    benchmark_name: str,
    task_id: str,
    solution: Any,
    ground_truth: Any,
    source_fingerprint: str,
    parse_function: Callable[[str], Any] | None = None,
    verify_function: Callable[[Any, Any], bool] | None = None,
    version_string: str | None = None,
) -> EvaluationRecord:
    """Extract and grade one answer while distinguishing model and grader failures."""

    current_version = version_string or grader_version()
    base = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "grader_version": current_version,
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "task_id": task_id,
    }
    if solution is None or not str(solution).strip():
        return EvaluationRecord(
            **base,
            status="incorrect",
            score=False,
            extracted_prediction=None,
            extracted_ground_truth=None,
            reason="missing_answer",
        )
    if ground_truth is None or not str(ground_truth).strip():
        return EvaluationRecord(
            **base,
            status="unresolved",
            score=None,
            extracted_prediction=None,
            extracted_ground_truth=None,
            reason="missing_ground_truth",
        )

    if parse_function is None or verify_function is None:
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            return EvaluationRecord(
                **base,
                status="unresolved",
                score=None,
                extracted_prediction=None,
                extracted_ground_truth=None,
                reason=f"grader_unavailable: {exc}",
            )
        parse_function = parse
        verify_function = verify

    try:
        parsed_ground_truth = parse_function(str(ground_truth))
    except Exception as exc:
        return EvaluationRecord(
            **base,
            status="unresolved",
            score=None,
            extracted_prediction=None,
            extracted_ground_truth=None,
            reason=f"ground_truth_parse_error: {type(exc).__name__}: {exc}",
        )
    if not parsed_ground_truth:
        return EvaluationRecord(
            **base,
            status="unresolved",
            score=None,
            extracted_prediction=None,
            extracted_ground_truth=None,
            reason="ground_truth_not_extracted",
        )

    try:
        parsed_prediction = parse_function(str(solution))
    except Exception as exc:
        return EvaluationRecord(
            **base,
            status="unresolved",
            score=None,
            extracted_prediction=None,
            extracted_ground_truth=_short_repr(parsed_ground_truth),
            reason=f"prediction_parse_error: {type(exc).__name__}: {exc}",
        )
    if not parsed_prediction:
        return EvaluationRecord(
            **base,
            status="incorrect",
            score=False,
            extracted_prediction=None,
            extracted_ground_truth=_short_repr(parsed_ground_truth),
            reason="answer_not_extracted",
        )

    try:
        score = bool(verify_function(parsed_ground_truth, parsed_prediction))
    except Exception as exc:
        return EvaluationRecord(
            **base,
            status="unresolved",
            score=None,
            extracted_prediction=_short_repr(parsed_prediction),
            extracted_ground_truth=_short_repr(parsed_ground_truth),
            reason=f"verification_error: {type(exc).__name__}: {exc}",
        )
    return EvaluationRecord(
        **base,
        status="correct" if score else "incorrect",
        score=score,
        extracted_prediction=_short_repr(parsed_prediction),
        extracted_ground_truth=_short_repr(parsed_ground_truth),
        reason=None,
    )


def _grading_worker(connection: Any) -> None:
    """Grade tasks in an isolated process that the parent can terminate safely."""

    import math_verify.grader as math_verify_grader
    from math_verify import parse, verify

    math_verify_grader.TIMEOUT_WARNING_SHOWN = True

    def parse_with_timeout(value: str) -> Any:
        return parse(value, parsing_timeout=5)

    # The parent enforces a hard timeout for the entire task. Disabling the nested
    # comparison alarm avoids a long Cartesian product of separately timed attempts.
    def verify_without_nested_timeout(gold: Any, prediction: Any) -> bool:
        return verify(gold, prediction, timeout_seconds=None)

    try:
        while True:
            payload = connection.recv()
            if payload is None:
                return
            record = grade_answer(
                **payload,
                parse_function=parse_with_timeout,
                verify_function=verify_without_nested_timeout,
            )
            connection.send(asdict(record))
    finally:
        connection.close()


class Grader(Protocol):
    """Minimal interface used by the resumable evaluation loop."""

    def grade(self, **payload: Any) -> EvaluationRecord: ...

    def close(self) -> None: ...


class IsolatedGrader:
    """Persistent subprocess grader with a hard wall-clock limit per task."""

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
        worker_target: Callable[[Any], None] = _grading_worker,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.worker_target = worker_target
        self._context = multiprocessing.get_context("spawn")
        self._connection: Any | None = None
        self._process: multiprocessing.Process | None = None

    def _start(self) -> None:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=self.worker_target, args=(child,), daemon=True
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process

    def _stop(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        if connection is not None:
            connection.close()

    def close(self) -> None:
        self._stop()

    def grade(self, **payload: Any) -> EvaluationRecord:
        if self._process is None or not self._process.is_alive():
            self._stop()
            self._start()
        assert self._connection is not None
        assert self._process is not None
        try:
            self._connection.send(payload)
            if self._connection.poll(self.timeout_seconds):
                raw = self._connection.recv()
                return EvaluationRecord(**raw)
        except (BrokenPipeError, EOFError, OSError):
            reason = "grader_worker_failed"
        else:
            reason = f"grader_timeout_after_{self.timeout_seconds:g}s"
        self._stop()
        return EvaluationRecord(
            schema_version=EVALUATION_SCHEMA_VERSION,
            source_fingerprint=str(payload["source_fingerprint"]),
            grader_version=str(payload["version_string"]),
            run_id=str(payload["run_id"]),
            benchmark_name=str(payload["benchmark_name"]),
            task_id=str(payload["task_id"]),
            status="unresolved",
            score=None,
            extracted_prediction=None,
            extracted_ground_truth=None,
            reason=reason,
        )


def write_evaluations_atomic(path: Path, records: Iterable[EvaluationRecord]) -> None:
    """Atomically write a complete sidecar so interrupted grading is never reused."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_evaluations(
    path: Path,
    *,
    artifact: BenchmarkArtifact | None = None,
    source_fingerprint: str | None = None,
    version_string: str | None = None,
) -> list[EvaluationRecord]:
    """Load a sidecar and validate its identity, schema, and unique task keys."""

    records: list[EvaluationRecord] = []
    seen: set[tuple[str, str]] = set()
    for raw in iter_jsonl(path):
        try:
            record = EvaluationRecord(**raw)
        except (TypeError, KeyError) as exc:
            raise ArtifactError(f"Invalid evaluation record in {path}: {exc}") from exc
        if record.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ArtifactError(f"Unsupported evaluation schema in {path}")
        if artifact and (
            record.run_id != artifact.run_id
            or record.benchmark_name != artifact.benchmark_name
        ):
            raise ArtifactError(f"Evaluation identity mismatch in {path}")
        if source_fingerprint and record.source_fingerprint != source_fingerprint:
            raise ArtifactError(f"Stale evaluation source fingerprint in {path}")
        if version_string and record.grader_version != version_string:
            raise ArtifactError(f"Stale evaluation grader version in {path}")
        key = (record.benchmark_name, record.task_id)
        if key in seen:
            raise ArtifactError(f"Duplicate evaluation task {key!r} in {path}")
        seen.add(key)
        records.append(record)
    return records


def evaluate_artifact(
    artifact: BenchmarkArtifact,
    evaluations_dir: Path,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
    timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    grader_factory: Callable[[], Grader] | None = None,
) -> tuple[Path, list[EvaluationRecord], bool]:
    """Grade one benchmark artifact, reusing only an exact content-addressed cache."""

    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    fingerprint = file_sha256(artifact.path)
    current_version = grader_version()
    cache_path = evaluation_cache_path(
        evaluations_dir, artifact, fingerprint, current_version
    )
    partial_path = cache_path.with_suffix(".partial.jsonl")
    tasks = load_task_records(artifact.path)
    expected_task_ids = [str(task["task_id"]) for task in tasks]
    expected_task_id_set = set(expected_task_ids)
    expected_keys = {
        (artifact.benchmark_name, task_id) for task_id in expected_task_ids
    }
    if cache_path.is_file() and not force:
        cached = load_evaluations(
            cache_path,
            artifact=artifact,
            source_fingerprint=fingerprint,
            version_string=current_version,
        )
        cached_keys = {
            (record.benchmark_name, record.task_id) for record in cached
        }
        if cached_keys == expected_keys:
            return cache_path, cached, True

    evaluated_by_id: dict[str, EvaluationRecord] = {}
    if partial_path.is_file() and not force:
        partial = load_evaluations(
            partial_path,
            artifact=artifact,
            source_fingerprint=fingerprint,
            version_string=current_version,
        )
        evaluated_by_id = {
            record.task_id: record
            for record in partial
            if record.task_id in expected_task_id_set
        }

    total = len(tasks)
    completed = len(evaluated_by_id)
    if progress:
        progress(completed, total)
    grader = (
        grader_factory()
        if grader_factory is not None
        else IsolatedGrader(timeout_seconds=timeout_seconds)
    )
    try:
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in evaluated_by_id:
                continue
            record = grader.grade(
                run_id=artifact.run_id,
                benchmark_name=artifact.benchmark_name,
                task_id=task_id,
                solution=task.get("solution"),
                ground_truth=task.get("ground_truth"),
                source_fingerprint=fingerprint,
                version_string=current_version,
            )
            evaluated_by_id[task_id] = record
            completed += 1
            if completed % checkpoint_every == 0:
                ordered = [
                    evaluated_by_id[item]
                    for item in expected_task_ids
                    if item in evaluated_by_id
                ]
                write_evaluations_atomic(partial_path, ordered)
            if progress:
                progress(completed, total)
    except BaseException:
        ordered = [
            evaluated_by_id[item]
            for item in expected_task_ids
            if item in evaluated_by_id
        ]
        write_evaluations_atomic(partial_path, ordered)
        raise
    finally:
        grader.close()

    evaluated = [evaluated_by_id[task_id] for task_id in expected_task_ids]
    write_evaluations_atomic(partial_path, evaluated)
    partial_path.replace(cache_path)
    return cache_path, evaluated, False
