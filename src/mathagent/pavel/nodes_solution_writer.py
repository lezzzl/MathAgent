import json
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from mathagent.pavel.nodes import (
    add_reasoning,
    add_usage,
    get_message_usage,
    load_prompt_role,
)
from mathagent.tools.python_tools import compact_stdout


def execution_succeeded(execution: dict[str, Any]) -> bool:
    """Проверяет, что notebook-вызов завершился без ошибки или timeout."""
    return execution.get("returncode") == 0 and not execution.get("timeout", False)


def recovered_execution(
    tool_call: dict[str, Any],
    repair_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Находит итоговый успешный execution автоматического repair-loop."""
    if not tool_call.get("recovered"):
        return None
    for repair_index in reversed(tool_call.get("repair_attempt_indices", [])):
        if not isinstance(repair_index, int) or not 0 <= repair_index < len(
            repair_history
        ):
            continue
        execution = repair_history[repair_index].get("execution")
        if isinstance(execution, dict) and execution_succeeded(execution):
            return execution
    return None


def build_solution_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Собирает только успешные компактные результаты CoT и execution tools."""
    evidence: list[dict[str, Any]] = []
    repair_history = list(state.get("repair_history", []))
    for tool_call in state.get("tool_history", []):
        tool_name = tool_call.get("tool_name")
        if tool_name == "cot":
            if not tool_call.get("success"):
                continue
            evidence.append(
                {
                    "tool_name": "cot",
                    "goal": tool_call.get("goal", ""),
                    "context": tool_call.get("context", ""),
                    "result": tool_call.get("result", ""),
                    "derivation": tool_call.get("derivation", ""),
                }
            )
            continue

        if tool_name not in {"python", "sympy"}:
            continue
        execution = tool_call.get("execution")
        if not isinstance(execution, dict) or not execution_succeeded(execution):
            execution = recovered_execution(tool_call, repair_history)
        if execution is None or not execution_succeeded(execution):
            continue
        evidence.append(
            {
                "tool_name": tool_name,
                "context": tool_call.get("context", ""),
                "stdout": compact_stdout(str(execution.get("stdout", ""))),
            }
        )
    return evidence


def build_previous_rejection(state: dict[str, Any]) -> dict[str, Any] | None:
    """Возвращает последнюю отклонённую версию и замечания verifier."""
    attempts = state.get("solution_attempts", [])
    if not attempts:
        return None
    latest = attempts[-1]
    verification = latest.get("verification")
    if not isinstance(verification, dict) or verification.get(
        "is_correct_solution"
    ):
        return None
    steps = verification.get("steps", [])
    failed_steps = [
        step
        for step in steps
        if isinstance(step, dict) and not step.get("is_ok", False)
    ]
    return {
        "rejected_answer": latest.get("answer", ""),
        "rejected_solution": latest.get("solution", ""),
        "failed_steps": failed_steps,
        "feedback_for_solver": verification.get("feedback_for_solver", ""),
        "analysis_thoughts": verification.get("analysis_thoughts", ""),
    }


def create_solution_writer_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт полное решение из ответа ReAct и компактных tool-evidence."""
    prompt_version, role = load_prompt_role(prompt_path, "solution_writer")

    def solution_writer(state: dict[str, Any]) -> dict[str, Any]:
        answer = state.get("proposed_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Solution writer requires a non-empty proposed answer")

        replacements = {
            "problem": state["problem"],
            "answer": answer.strip(),
            "plan": str(state.get("plan", "") or "No advisory plan was generated."),
            "evidence": json.dumps(
                build_solution_evidence(state),
                ensure_ascii=False,
                indent=2,
            ),
            "previous_rejection": json.dumps(
                build_previous_rejection(state),
                ensure_ascii=False,
                indent=2,
            ),
        }
        task_prompt = role["task"]
        for field, value in replacements.items():
            task_prompt = task_prompt.replace(f"{{{field}}}", value)

        writer_history = list(state.get("solution_writer_history", []))
        writer_index = len(writer_history)
        call_key = f"solution_writer_{writer_index}"
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=role["system"]),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = time.perf_counter() - started
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("Solution writer returned an empty solution")

        usage = get_message_usage(message)
        writer_history.append(
            {
                "index": writer_index,
                "usage": usage,
                "latency_seconds": round(latency, 3),
                "finish_reason": usage.get("finish_reason"),
            }
        )
        return {
            "proposed_solution": message.content.strip(),
            "solution_writer_history": writer_history,
            "reasoning": add_reasoning(state, call_key, message),
            "usage": add_usage(state, call_key, message),
            "prompt_version": prompt_version,
            "status": "verification_requested",
        }

    return solution_writer
