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


def truncate_tool_output(value: str, limit: int) -> str:
    """Оставляет начало и конец длинного вывода для контекста модели."""
    if len(value) <= limit:
        return value

    omitted = len(value) - limit
    marker = ""
    for _ in range(3):
        marker = f"\n... [truncated {omitted} characters] ...\n"
        omitted = len(value) - (limit - len(marker))
    marker = f"\n... [truncated {omitted} characters] ...\n"
    if len(marker) >= limit:
        return value[:limit]

    available = limit - len(marker)
    head_length = available // 2
    tail_length = available - head_length
    return f"{value[:head_length]}{marker}{value[-tail_length:]}"


def create_python_tool(timeout: float, output_limit_chars: int) -> BaseTool:
    """Создаёт Python tool с полным artifact и сокращённым ответом для LLM."""
    executor = PythonExecutor(timeout=timeout)

    @tool("python", response_format="content_and_artifact")
    def python_tool(code: str, purpose: str) -> tuple[str, dict[str, Any]]:
        """Execute self-contained Python/SymPy code for a mathematical purpose."""
        del purpose
        started = time.perf_counter()
        execution = executor.run(code).to_dict()
        execution["latency_seconds"] = round(time.perf_counter() - started, 3)
        model_execution = {
            **execution,
            "stdout": truncate_tool_output(execution["stdout"], output_limit_chars),
            "stderr": truncate_tool_output(execution["stderr"], output_limit_chars),
        }
        return json.dumps(model_execution, ensure_ascii=False), execution

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


def format_retry_messages(
    message: AIMessage,
    clean_message: AIMessage,
) -> list[Any]:
    """Закрывает невалидные tool calls и требует терминальный final_answer."""
    retry_message = clean_message
    if message.invalid_tool_calls:
        retry_message = AIMessage(
            content=message.content,
            id=clean_message.id,
        )
    retry_messages: list[Any] = [retry_message]
    for tool_call in message.tool_calls:
        retry_messages.append(
            ToolMessage(
                content=(
                    "This tool call is invalid and was not executed. "
                    "Submit the best supported answer with final_answer."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
                id=f"react:format_error:{tool_call['id']}",
            )
        )
    retry_messages.append(
        HumanMessage(
            content=(
                "Your previous response did not follow the required tool protocol. "
                "Do not call Python. Submit the best answer supported by the "
                "available reasoning and tool results using final_answer."
            ),
            id="react:format_retry",
        )
    )
    return retry_messages


def create_react_agent_node(
    model: Any,
    python_tool: BaseTool,
    final_answer_tool: BaseTool,
    prompt_path: Path,
    max_tool_calls: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Вызывает ReAct-модель и обрабатывает Python либо final_answer action."""
    prompt_version, role = load_prompt_role(prompt_path, "agent")
    model_with_tools = model.bind_tools(
        [python_tool, final_answer_tool],
        tool_choice="auto",
        strict=True,
        parallel_tool_calls=False,
    )
    model_with_final_answer = model.bind_tools(
        [final_answer_tool],
        tool_choice=final_answer_tool.name,
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
        forced_final_reason = state.get("forced_final_reason")
        if tool_call_count >= max_tool_calls and forced_final_reason is None:
            forced_final_reason = "tool_limit_reached"
            new_messages.append(
                HumanMessage(
                    content=(
                        "The Python tool-call limit has been reached. Do not call "
                        "Python. Submit the best answer supported by the available "
                        "reasoning and tool results using final_answer."
                    ),
                    id="react:tool_limit",
                )
            )

        forced_final = forced_final_reason is not None
        invocation_messages = [*messages, *new_messages]
        iteration = len(state.get("agent_history", []))
        started = time.perf_counter()
        message = (
            model_with_final_answer.invoke(invocation_messages)
            if forced_final
            else model_with_tools.invoke(invocation_messages)
        )
        latency = time.perf_counter() - started

        call_key = f"agent_{iteration}"
        clean_message = clean_agent_message(message, f"react:agent:{iteration}")
        agent_history = list(state.get("agent_history", []))
        history_entry: dict[str, Any] = {
            "iteration": iteration,
            "mode": "forced_final" if forced_final else "tools_enabled",
            "action": "invalid_format",
            "usage": get_message_usage(message),
            "latency_seconds": round(latency, 3),
        }
        base_update: dict[str, Any] = {
            "reasoning": add_reasoning(state, call_key, message),
            "usage": add_usage(state, call_key, message),
            "prompt_version": prompt_version,
        }

        format_error: str | None = None
        if message.invalid_tool_calls:
            format_error = "ReAct agent returned an invalid tool call"
        elif len(message.tool_calls) != 1:
            format_error = "ReAct agent must return exactly one tool call"
        else:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.get("name")
            arguments = tool_call.get("args") or {}
            history_entry["tool_call_id"] = tool_call.get("id")
            if tool_name == python_tool.name:
                code = arguments.get("code")
                purpose = arguments.get("purpose")
                if forced_final:
                    format_error = "Python cannot be called during forced finalization"
                elif not isinstance(code, str) or not code.strip():
                    format_error = "Python tool call must contain non-empty code"
                elif not isinstance(purpose, str) or not purpose.strip():
                    format_error = "Python tool call must contain non-empty purpose"
                else:
                    history_entry["action"] = "python"
                    return {
                        **base_update,
                        "messages": [*new_messages, clean_message],
                        "agent_history": [*agent_history, history_entry],
                        "status": "tool_requested",
                    }
            elif tool_name == final_answer_tool.name:
                answer = arguments.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    format_error = "final_answer must contain a non-empty answer"
                else:
                    finish_reason = forced_final_reason or "final_answer"
                    history_entry["action"] = "final_answer"
                    completed_history = [*agent_history, history_entry]
                    return {
                        **base_update,
                        "messages": [*new_messages, clean_message],
                        "agent_history": completed_history,
                        "solution": answer.strip(),
                        "status": "completed",
                        "finish_reason": finish_reason,
                        "trace": build_react_trace(
                            completed_history,
                            list(state.get("tool_history", [])),
                            finish_reason,
                        ),
                    }
            else:
                format_error = f"ReAct agent requested unknown tool: {tool_name}"

        if state.get("format_retry_count", 0) >= 1:
            raise ValueError(format_error or "ReAct agent violated tool protocol")

        history_entry["error"] = format_error
        return {
            **base_update,
            "messages": [
                *new_messages,
                *format_retry_messages(message, clean_message),
            ],
            "agent_history": [*agent_history, history_entry],
            "format_retry_count": 1,
            "forced_final_reason": "format_recovery",
            "status": "format_retry",
        }

    return agent


def route_after_react_agent(state: dict[str, Any]) -> str:
    """Направляет Python в ToolNode, format retry обратно в agent, ответ в END."""
    status = state["status"]
    if status == "tool_requested":
        return "tools"
    if status == "format_retry":
        return "agent"
    if status == "completed":
        return "finished"
    raise ValueError(f"Unknown ReAct agent status: {status}")


def record_python_tool_call(state: dict[str, Any]) -> dict[str, Any]:
    """Сохраняет полный artifact Python после выполнения стандартной ToolNode."""
    messages = state["messages"]
    if len(messages) < 2:
        raise ValueError("Python tool result has no matching agent message")

    agent_message = messages[-2]
    tool_message = messages[-1]
    if not isinstance(agent_message, AIMessage) or len(agent_message.tool_calls) != 1:
        raise ValueError("Python tool result requires exactly one AI tool call")
    if not isinstance(tool_message, ToolMessage):
        raise ValueError("Python tool execution did not return a ToolMessage")

    tool_call = agent_message.tool_calls[0]
    if tool_message.tool_call_id != tool_call["id"]:
        raise ValueError("Python ToolMessage does not match the requested tool call")
    if not isinstance(tool_message.artifact, dict):
        raise ValueError("Python ToolMessage has no full execution artifact")

    arguments = tool_call["args"]
    tool_index = state.get("tool_call_count", 0)
    tool_history = list(state.get("tool_history", []))
    tool_history.append(
        {
            "index": tool_index,
            "tool_call_id": tool_call["id"],
            "purpose": arguments["purpose"].strip(),
            "code": arguments["code"].strip(),
            "execution": tool_message.artifact,
        }
    )
    return {
        "tool_call_count": tool_index + 1,
        "tool_history": tool_history,
        "status": "tool_executed",
    }
