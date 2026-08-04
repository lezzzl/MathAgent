import json
import re
import time
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from mathagent.pipelines.agent_eval.nodes import (
    get_message_reasoning,
    get_message_usage,
    load_prompt_role,
)


FENCED_JSON_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalized_cot_json(value: Any) -> dict[str, str] | None:
    """Нормализует JSON CoT, если он содержит непустой строковый result."""
    if not isinstance(value, dict):
        return None
    result = value.get("result")
    if not isinstance(result, str) or not result.strip():
        return None
    derivation = value.get("derivation", "")
    if not isinstance(derivation, str):
        derivation = ""
    return {
        "derivation": derivation.strip(),
        "result": result.strip(),
    }


def parse_cot_output(raw_output: str) -> dict[str, Any]:
    """Разбирает CoT как JSON, fenced JSON, embedded JSON или plain text."""
    text = raw_output.strip()
    if not text:
        return {
            "derivation": "",
            "result": "",
            "parse_mode": None,
            "parse_warning": "CoT subagent output is empty",
        }

    try:
        exact_value = json.loads(text)
    except json.JSONDecodeError:
        exact_value = None
    normalized = normalized_cot_json(exact_value)
    if normalized is not None:
        return {
            **normalized,
            "parse_mode": "json",
            "parse_warning": None,
        }

    for match in FENCED_JSON_PATTERN.finditer(text):
        try:
            fenced_value = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        normalized = normalized_cot_json(fenced_value)
        if normalized is not None:
            return {
                **normalized,
                "parse_mode": "fenced_json",
                "parse_warning": "CoT JSON was wrapped in a Markdown code fence",
            }

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            embedded_value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        normalized = normalized_cot_json(embedded_value)
        if normalized is not None:
            return {
                **normalized,
                "parse_mode": "embedded_json",
                "parse_warning": "CoT JSON was surrounded by additional text",
            }

    return {
        "derivation": "",
        "result": text,
        "parse_mode": "plain_text",
        "parse_warning": "CoT output was accepted as plain text",
    }


def create_cot_tool(model: Any, prompt_path: Path) -> BaseTool:
    """Создаёт LLM-tool для решения одной локальной математической цели."""
    _, role = load_prompt_role(prompt_path, "cot")
    system_prompt = role.get("system")
    task_template = role.get("task")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("CoT prompt role must contain a non-empty system prompt")
    if not isinstance(task_template, str) or not task_template.strip():
        raise ValueError("CoT prompt role must contain a non-empty task prompt")

    @tool("cot", response_format="content_and_artifact")
    def cot_tool(
        goal: str,
        context: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> tuple[str, dict[str, Any]]:
        """Solve one focused mathematical subgoal from the original problem.

        Provide a precise local goal and a compact context containing the
        mathematical facts, assumptions, reductions, and approach established so
        far. The tool automatically sees the original problem but not the full
        ReAct history. It returns a concise justification and the result
        established for that goal. It has no access to Python or other tools.
        """
        problem = state.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("CoT tool requires the original problem in graph state")

        task_prompt = task_template.replace("{problem}", problem).replace(
            "{goal}", goal
        ).replace("{context}", context)
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = round(time.perf_counter() - started, 3)
        raw_output = (
            message.content.strip() if isinstance(message.content, str) else ""
        )
        parsed = parse_cot_output(raw_output)
        derivation = parsed["derivation"]
        result = parsed["result"]
        parse_mode = parsed["parse_mode"]
        parse_warning = parsed["parse_warning"]
        success = bool(result)
        error = None if success else "CoT subagent output is empty"
        artifact = {
            "goal": goal.strip(),
            "context": context.strip(),
            "derivation": derivation,
            "result": result,
            "success": success,
            "error": error,
            "parse_mode": parse_mode,
            "parse_warning": parse_warning,
            "latency_seconds": latency,
            "usage": get_message_usage(message),
            "reasoning": get_message_reasoning(message),
        }
        if parse_mode != "json":
            artifact["raw_output"] = raw_output
        if not success:
            return json.dumps(
                {
                    "error": "invalid_cot_output",
                    "message": error,
                },
                ensure_ascii=False,
            ), artifact
        return json.dumps(
            {
                "derivation": derivation,
                "result": result,
            },
            ensure_ascii=False,
        ), artifact

    return cot_tool
