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
from mathagent.tools.final_answer import create_derived_solution_tool
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
    protocol = role.get("protocol")
    structured_output = protocol == "independent-structured-v2"
    writer_tool = create_derived_solution_tool() if structured_output else None
    model_with_output = (
        model.bind_tools(
            [writer_tool],
            tool_choice=writer_tool.name,
            strict=True,
            parallel_tool_calls=False,
        )
        if writer_tool is not None
        else None
    )

    def parse_structured_output(message: Any) -> tuple[str | None, str]:
        """Проверяет обязательный submit_solution tool writer-ноды."""
        if message.invalid_tool_calls or len(message.tool_calls) != 1:
            raise ValueError("Solution writer must return exactly one tool call")
        tool_call = message.tool_calls[0]
        if writer_tool is None or tool_call.get("name") != writer_tool.name:
            raise ValueError("Solution writer must call submit_solution")
        arguments = tool_call.get("args") or {}
        solution = arguments.get("solution")
        derived_answer = arguments.get("derived_answer")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("Solution writer must return a non-empty solution")
        if derived_answer is not None and (
            not isinstance(derived_answer, str) or not derived_answer.strip()
        ):
            raise ValueError("derived_answer must be non-empty text or null")
        return (
            derived_answer.strip() if isinstance(derived_answer, str) else None,
            solution.strip(),
        )

    def solution_writer(state: dict[str, Any]) -> dict[str, Any]:
        answer = state.get("proposed_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Solution writer requires a non-empty proposed answer")

        replacements = {
            "problem": state["problem"],
            "answer": answer.strip(),
            "solution_context": str(state.get("proposed_solution_context", "")),
            "plan": str(state.get("plan", "") or "No advisory plan was generated."),
            "evidence": json.dumps(
                build_solution_evidence(state),
                ensure_ascii=False,
                indent=2,
            ),
        }
        if not structured_output:
            replacements["previous_rejection"] = json.dumps(
                build_previous_rejection(state),
                ensure_ascii=False,
                indent=2,
            )
        elif not replacements["solution_context"].strip():
            raise ValueError(
                "Independent solution writer requires a non-empty solution_context"
            )
        task_prompt = role["task"]
        for field, value in replacements.items():
            task_prompt = task_prompt.replace(f"{{{field}}}", value)

        writer_history = list(state.get("solution_writer_history", []))
        writer_index = len(writer_history)
        call_key = f"solution_writer_{writer_index}"
        base_messages = [
            SystemMessage(content=role["system"]),
            HumanMessage(content=task_prompt),
        ]
        total_latency = 0.0
        usage_by_attempt: dict[str, Any] = {}
        reasoning = dict(state.get("reasoning", {}))
        usage_state = dict(state.get("usage", {}))
        derived_answer: str | None = None
        solution = ""
        accepted_message: Any = None
        format_retries = 0
        attempts = 2 if structured_output else 1
        for attempt_index in range(attempts):
            messages = list(base_messages)
            if attempt_index:
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous response violated the output contract. "
                            "Call submit_solution exactly once with derived_answer "
                            "(a non-empty string or null) and a non-empty solution."
                        )
                    )
                )
            started = time.perf_counter()
            message = (
                model_with_output.invoke(messages)
                if model_with_output is not None
                else model.invoke(messages)
            )
            total_latency += time.perf_counter() - started
            attempt_key = (
                call_key
                if attempt_index == 0
                else f"{call_key}_format_retry"
            )
            usage_by_attempt[f"attempt_{attempt_index}"] = get_message_usage(
                message
            )
            usage_state = add_usage(
                {**state, "usage": usage_state}, attempt_key, message
            )
            reasoning = add_reasoning(
                {**state, "reasoning": reasoning}, attempt_key, message
            )
            if not structured_output:
                if not isinstance(message.content, str) or not message.content.strip():
                    raise ValueError("Solution writer returned an empty solution")
                solution = message.content.strip()
                accepted_message = message
                break
            try:
                derived_answer, solution = parse_structured_output(message)
            except ValueError:
                if attempt_index == 0:
                    format_retries = 1
                    continue
                raise
            accepted_message = message
            break

        if accepted_message is None:
            raise ValueError("Solution writer did not produce a valid solution")
        accepted_usage = get_message_usage(accepted_message)
        writer_history.append(
            {
                "index": writer_index,
                "usage": (
                    usage_by_attempt
                    if structured_output and format_retries
                    else accepted_usage
                ),
                "latency_seconds": round(total_latency, 3),
                "finish_reason": accepted_usage.get("finish_reason"),
                **(
                    {"format_retries": format_retries}
                    if structured_output
                    else {}
                ),
            }
        )
        return {
            "proposed_solution": solution,
            **({"derived_answer": derived_answer} if structured_output else {}),
            "solution_writer_history": writer_history,
            "reasoning": reasoning,
            "usage": usage_state,
            "prompt_version": prompt_version,
            "status": "verification_requested",
        }

    return solution_writer
