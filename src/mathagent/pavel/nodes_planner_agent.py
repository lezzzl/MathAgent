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


def render_known_placeholders(template: str, **values: str) -> str:
    """Подставляет только известные поля, не интерпретируя LaTeX-скобки"""
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{name}}}", value)
    return rendered


def create_planner_agent_planner_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт один advisory plan перед обычным вызовом решающей модели"""
    prompt_version, role = load_prompt_role(prompt_path, "planner")

    def planner(state: dict[str, Any]) -> dict[str, Any]:
        task_prompt = render_known_placeholders(
            role["task"],
            problem=state["problem"],
        )
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=role["system"]),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = time.perf_counter() - started
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("Planner agent returned an empty plan")

        usage = get_message_usage(message)
        return {
            "plan": message.content.strip(),
            "planner_trace": {
                "latency_seconds": round(latency, 3),
                "finish_reason": usage.get("finish_reason"),
            },
            "reasoning": add_reasoning(state, "planner", message),
            "usage": add_usage(state, "planner", message),
            "prompt_version": prompt_version,
        }

    return planner


def create_planner_agent_solver_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Решает задачу по advisory plan одним обычным LLM-вызовом без tools"""
    prompt_version, role = load_prompt_role(prompt_path, "agent")

    def agent(state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan")
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("Planner agent requires a non-empty advisory plan")

        task_prompt = render_known_placeholders(
            role["task"],
            problem=state["problem"],
            plan=plan,
        )
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=role["system"]),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = time.perf_counter() - started
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("Planner agent returned an empty solution")

        usage = get_message_usage(message)
        return {
            "solution": message.content.strip(),
            "reasoning": add_reasoning(state, "agent", message),
            "usage": add_usage(state, "agent", message),
            "prompt_version": prompt_version,
            "trace": {
                "status": "completed",
                "plan": plan,
                "nodes": {
                    "planner": dict(state.get("planner_trace", {})),
                    "agent": {
                        "latency_seconds": round(latency, 3),
                        "finish_reason": usage.get("finish_reason"),
                    },
                },
            },
        }

    return agent
