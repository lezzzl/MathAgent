"""Answer grading used while publishing source experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Grade:
    status: str
    correct: bool | None
    reason: str | None = None


def grade_answer(
    solution: str | None,
    ground_truth: str | None,
    *,
    parse: Callable[[str], Any] | None = None,
    verify: Callable[[Any, Any], bool] | None = None,
) -> Grade:
    """Grade a mathematical answer and keep grader failures unresolved."""

    if solution is None or not solution.strip():
        return Grade("incorrect", False, "missing_answer")
    if ground_truth is None or not ground_truth.strip():
        return Grade("unresolved", None, "missing_ground_truth")
    if parse is None or verify is None:
        from math_verify import parse as math_parse
        from math_verify import verify as math_verify

        parse, verify = math_parse, math_verify
    try:
        expected = parse(ground_truth)
    except Exception as exc:
        return Grade("unresolved", None, f"ground_truth_parse_error:{type(exc).__name__}")
    if not expected:
        return Grade("unresolved", None, "ground_truth_not_extracted")
    try:
        prediction = parse(solution)
    except Exception as exc:
        return Grade("incorrect", False, f"prediction_parse_error:{type(exc).__name__}")
    if not prediction:
        return Grade("incorrect", False, "answer_not_extracted")
    try:
        correct = bool(verify(expected, prediction))
    except Exception as exc:
        return Grade("unresolved", None, f"verification_error:{type(exc).__name__}")
    return Grade("correct" if correct else "incorrect", correct)
