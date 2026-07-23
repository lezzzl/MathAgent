from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dashboard.artifacts import (
    ArtifactError,
    BenchmarkArtifact,
    evaluation_cache_path,
    file_sha256,
)
from dashboard.evaluation import (
    EvaluationRecord,
    IsolatedGrader,
    evaluate_artifact,
    grade_answer,
    grader_version,
    load_evaluations,
    write_evaluations_atomic,
)


def _record(**overrides: object) -> EvaluationRecord:
    values = {
        "schema_version": 1,
        "source_fingerprint": "abc",
        "grader_version": "grader-1",
        "run_id": "run",
        "benchmark_name": "AIME",
        "task_id": "1",
        "status": "correct",
        "score": True,
        "extracted_prediction": "2",
        "extracted_ground_truth": "2",
        "reason": None,
    }
    values.update(overrides)
    return EvaluationRecord(**values)


def test_grade_answer_uses_injected_math_grader() -> None:
    result = grade_answer(
        run_id="run",
        benchmark_name="AIME",
        task_id="1",
        solution=r"\boxed{\frac{1}{2}}",
        ground_truth=r"\frac{1}{2}",
        source_fingerprint="abc",
        parse_function=lambda value: value.replace(r"\boxed{", "").rstrip("}"),
        verify_function=lambda gold, answer: gold == answer,
        version_string="test-grader",
    )

    assert result.status == "correct"
    assert result.score is True


def test_missing_prediction_is_incorrect_without_calling_grader() -> None:
    result = grade_answer(
        run_id="run",
        benchmark_name="AIME",
        task_id="1",
        solution=None,
        ground_truth="2",
        source_fingerprint="abc",
        parse_function=lambda _: pytest.fail("parser must not be called"),
        verify_function=lambda *_: pytest.fail("verifier must not be called"),
        version_string="test-grader",
    )

    assert result.status == "incorrect"
    assert result.score is False
    assert result.reason == "missing_answer"


def test_unparseable_prediction_is_incorrect() -> None:
    result = grade_answer(
        run_id="run",
        benchmark_name="AIME",
        task_id="1",
        solution="no final answer",
        ground_truth="2",
        source_fingerprint="abc",
        parse_function=lambda value: ["2"] if value == "2" else [],
        verify_function=lambda *_: True,
        version_string="test-grader",
    )

    assert result.score is False
    assert result.reason == "answer_not_extracted"


def test_ground_truth_parser_failure_is_unresolved() -> None:
    def fail(_: str) -> object:
        raise RuntimeError("parser crashed")

    result = grade_answer(
        run_id="run",
        benchmark_name="IMO",
        task_id="1",
        solution="2",
        ground_truth="bad",
        source_fingerprint="abc",
        parse_function=fail,
        verify_function=lambda *_: True,
        version_string="test-grader",
    )

    assert result.status == "unresolved"
    assert result.score is None
    assert result.reason and result.reason.startswith("ground_truth_parse_error")


def test_sidecar_validation_rejects_stale_source(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.jsonl"
    write_evaluations_atomic(path, [_record()])
    artifact = BenchmarkArtifact("run", "AIME", tmp_path / "aime.jsonl", {})

    with pytest.raises(ArtifactError, match="Stale evaluation"):
        load_evaluations(
            path,
            artifact=artifact,
            source_fingerprint="different",
            version_string="grader-1",
        )


def test_sidecar_validation_rejects_duplicate_tasks(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.jsonl"
    write_evaluations_atomic(path, [_record(), _record()])

    with pytest.raises(ArtifactError, match="Duplicate evaluation"):
        load_evaluations(path)


def _hanging_worker(connection: object) -> None:
    connection.recv()  # type: ignore[attr-defined]
    time.sleep(60)


def test_isolated_grader_enforces_hard_timeout() -> None:
    grader = IsolatedGrader(timeout_seconds=0.05, worker_target=_hanging_worker)
    try:
        result = grader.grade(
            run_id="run",
            benchmark_name="IMO",
            task_id="slow",
            solution="x",
            ground_truth="x",
            source_fingerprint="abc",
            version_string="test-grader",
        )
    finally:
        grader.close()

    assert result.status == "unresolved"
    assert result.reason == "grader_timeout_after_0.05s"


class _InterruptingGrader:
    def __init__(self, interrupt_after: int | None = None) -> None:
        self.calls = 0
        self.interrupt_after = interrupt_after
        self.closed = False

    def grade(self, **payload: object) -> EvaluationRecord:
        self.calls += 1
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise KeyboardInterrupt
        return EvaluationRecord(
            schema_version=1,
            source_fingerprint=str(payload["source_fingerprint"]),
            grader_version=str(payload["version_string"]),
            run_id=str(payload["run_id"]),
            benchmark_name=str(payload["benchmark_name"]),
            task_id=str(payload["task_id"]),
            status="correct",
            score=True,
            extracted_prediction="1",
            extracted_ground_truth="1",
            reason=None,
        )

    def close(self) -> None:
        self.closed = True


def test_interrupted_evaluation_checkpoints_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "run" / "benchmark.jsonl"
    output.parent.mkdir()
    output.write_text(
        "".join(
            json.dumps(
                {
                    "run_id": "run",
                    "benchmark_name": "Bench",
                    "task_id": str(task_id),
                    "solution": "1",
                    "ground_truth": "1",
                }
            )
            + "\n"
            for task_id in range(3)
        ),
        encoding="utf-8",
    )
    artifact = BenchmarkArtifact("run", "Bench", output, {})
    evaluations_dir = tmp_path / "evaluations"
    interrupted = _InterruptingGrader(interrupt_after=1)

    with pytest.raises(KeyboardInterrupt):
        evaluate_artifact(
            artifact,
            evaluations_dir,
            checkpoint_every=1,
            grader_factory=lambda: interrupted,
        )

    cache_path = evaluation_cache_path(
        evaluations_dir, artifact, file_sha256(output), grader_version()
    )
    partial_path = cache_path.with_suffix(".partial.jsonl")
    assert interrupted.closed
    assert len(load_evaluations(partial_path)) == 1
    assert not cache_path.exists()

    resumed = _InterruptingGrader()
    final_path, records, reused = evaluate_artifact(
        artifact,
        evaluations_dir,
        checkpoint_every=1,
        grader_factory=lambda: resumed,
    )

    assert reused is False
    assert resumed.calls == 2
    assert len(records) == 3
    assert final_path == cache_path
    assert cache_path.exists()
    assert not partial_path.exists()
