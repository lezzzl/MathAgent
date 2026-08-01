import json
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from mathagent.pipelines.agent_eval.nodes import (
    add_reasoning,
    add_usage,
    get_message_usage,
    load_prompt_role,
)
from mathagent.tools.python_executor import PythonExecutor


def create_python_tool(timeout: float) -> BaseTool:
    executor = PythonExecutor(timeout=timeout)

    @tool("python")
    def python_tool(code: str, purpose: str) -> str:
        """Execute self-contained Python code for a stated mathematical purpose."""
        del purpose
        return json.dumps(executor.run(code).to_dict(), ensure_ascii=False)

    return python_tool


def clean_agent_message(message: AIMessage, message_id: str) -> AIMessage:
    """Оставляет visible content и tool calls, но не hidden reasoning модели."""
    return AIMessage(
        content=message.content,
        tool_calls=message.tool_calls,
        invalid_tool_calls=message.invalid_tool_calls,
        id=message_id,
    )


def build_react_trace(
    agent_history: list[dict[str, Any]],
    tool_history: list[dict[str, Any]],
    finish_reason: str,
) -> dict[str, Any]:
    """Собирает компактную траекторию без messages и hidden reasoning."""
    return {
        "status": "completed",
        "finish_reason": finish_reason,
        "agent_calls": agent_history,
        "tool_calls": tool_history,
    }


def create_react_agent_node(
    model: Any,
    python_tool: BaseTool,
    prompt_path: Path,
    max_tool_calls: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Вызывает модель с Python tool либо без tools после исчерпания лимита."""
    prompt_version, role = load_prompt_role(prompt_path, "agent")
    model_with_tools = model.bind_tools(
        [python_tool],
        tool_choice="auto",
        strict=True,
        parallel_tool_calls=False,
    )

    def agent(state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        new_messages: list[Any] = []
        if not messages:
            new_messages = [
                SystemMessage(content=role["system"], id="react:system"),
                HumanMessage(
                    content=role["task"].replace("{problem}", state["problem"]),
                    id="react:problem",
                ),
            ]

        tool_call_count = state.get("tool_call_count", 0)
        forced_final = tool_call_count >= max_tool_calls
        if forced_final:
            new_messages.append(
                HumanMessage(
                    content=(
                        "The Python tool-call limit has been reached. Do not call "
                        "any tools. Return the best final answer supported by the "
                        "available reasoning and tool results in \\boxed{...} form."
                    ),
                    id="react:tool_limit",
                )
            )

        invocation_messages = [*messages, *new_messages]
        iteration = len(state.get("agent_history", []))
        started = time.perf_counter()
        message = (
            model.invoke(invocation_messages)
            if forced_final
            else model_with_tools.invoke(invocation_messages)
        )
        latency = time.perf_counter() - started

        if message.invalid_tool_calls:
            raise ValueError("ReAct agent returned an invalid Python tool call")
        if len(message.tool_calls) > 1:
            raise ValueError("ReAct agent returned more than one Python tool call")
        if forced_final and message.tool_calls:
            raise ValueError("ReAct agent called Python after the tool-call limit")

        call_key = f"agent_{iteration}"
        clean_message = clean_agent_message(message, f"react:agent:{iteration}")
        agent_history = list(state.get("agent_history", []))
        history_entry: dict[str, Any] = {
            "iteration": iteration,
            "mode": "forced_final" if forced_final else "tools_enabled",
            "action": "python" if message.tool_calls else "final",
            "usage": get_message_usage(message),
            "latency_seconds": round(latency, 3),
        }

        update: dict[str, Any] = {
            "messages": [*new_messages, clean_message],
            "reasoning": add_reasoning(state, call_key, message),
            "usage": add_usage(state, call_key, message),
            "prompt_version": prompt_version,
        }
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.get("name") != python_tool.name:
                raise ValueError(
                    f"ReAct agent requested unknown tool: {tool_call.get('name')}"
                )
            arguments = tool_call.get("args") or {}
            code = arguments.get("code")
            purpose = arguments.get("purpose")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Python tool call must contain non-empty code")
            if not isinstance(purpose, str) or not purpose.strip():
                raise ValueError("Python tool call must contain non-empty purpose")
            history_entry["tool_call_id"] = tool_call["id"]
            update["status"] = "tool_requested"
        else:
            if not isinstance(message.content, str) or not message.content.strip():
                raise ValueError("ReAct agent returned an empty final answer")
            finish_reason = (
                "tool_limit_reached" if forced_final else "model_finished"
            )
            update.update(
                {
                    "solution": message.content,
                    "status": "completed",
                    "finish_reason": finish_reason,
                    "trace": build_react_trace(
                        [*agent_history, history_entry],
                        list(state.get("tool_history", [])),
                        finish_reason,
                    ),
                }
            )

        update["agent_history"] = [*agent_history, history_entry]
        return update

    return agent


def create_python_tool_node(
    python_tool: BaseTool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Исполняет последний нативный tool call и возвращает ToolMessage модели."""

    def execute_python(state: dict[str, Any]) -> dict[str, Any]:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or len(last_message.tool_calls) != 1:
            raise ValueError("Python execution requires exactly one AI tool call")

        tool_call = last_message.tool_calls[0]
        arguments = tool_call["args"]
        started = time.perf_counter()
        result_text = python_tool.invoke(arguments)
        latency = time.perf_counter() - started
        execution = json.loads(result_text)
        tool_index = state.get("tool_call_count", 0)
        execution_trace = {
            **execution,
            "latency_seconds": round(latency, 3),
        }
        tool_history = list(state.get("tool_history", []))
        tool_history.append(
            {
                "index": tool_index,
                "tool_call_id": tool_call["id"],
                "purpose": arguments["purpose"].strip(),
                "code": arguments["code"].strip(),
                "execution": execution_trace,
            }
        )
        return {
            "messages": [
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call["id"],
                    name=python_tool.name,
                    id=f"react:tool:{tool_index}",
                )
            ],
            "tool_call_count": tool_index + 1,
            "tool_history": tool_history,
            "status": "tool_executed",
        }

    return execute_python
