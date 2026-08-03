import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from mathagent.pipelines.agent_eval.nodes import (
    add_reasoning,
    add_usage,
    get_message_usage,
    load_prompt_role,
)


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
    tool_calls_remaining: int,
    available_tool_names: list[str],
) -> list[Any]:
    """Закрывает невалидные tool calls и запрашивает одно корректное действие."""
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
                    "Return exactly one valid tool call."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
                id=f"react:format_error:{tool_call['id']}",
            )
        )

    if tool_calls_remaining > 0:
        choices = ", ".join(available_tool_names)
        allowed_tools = f"Use exactly one of these tools: {choices}."
    else:
        allowed_tools = "The tool budget is exhausted, so use final_answer."
    retry_messages.append(
        HumanMessage(
            content=(
                "Your previous response did not follow the required tool protocol. "
                f"Return exactly one valid tool call. {allowed_tools}"
            ),
            id="react:format_retry",
        )
    )
    return retry_messages


def execution_failed(tool_history: list[dict[str, Any]]) -> bool:
    """Проверяет, что последняя исполненная попытка завершилась ошибкой."""
    if not tool_history:
        return False
    execution = tool_history[-1]["execution"]
    return execution["returncode"] != 0 or execution["timeout"] is True


def create_react_agent_node(
    model: Any,
    python_tool: BaseTool,
    final_answer_tool: BaseTool,
    prompt_path: Path,
    max_tool_calls: int,
    repair_tool: BaseTool | None = None,
    cot_tool: BaseTool | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Вызывает ReAct-модель и обрабатывает tools выбранной версии промпта."""
    if repair_tool is not None and cot_tool is not None:
        raise ValueError("ReAct graph cannot enable cot and repair together")

    prompt_version, role = load_prompt_role(prompt_path, "agent")
    budgeted_tools = [python_tool]
    if cot_tool is not None:
        budgeted_tools.insert(0, cot_tool)
    if repair_tool is not None:
        budgeted_tools.append(repair_tool)
    available_tools = [*budgeted_tools, final_answer_tool]
    available_tool_names = [tool.name for tool in available_tools]
    model_with_tools = model.bind_tools(
        available_tools,
        tool_choice="auto",
        strict=False,
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
        tool_calls_remaining = max(max_tool_calls - tool_call_count, 0)
        forced_final = tool_calls_remaining == 0
        if forced_final:
            new_messages.append(
                HumanMessage(
                    content=(
                        "The tool-call limit has been reached. Do not call any "
                        "reasoning or execution tool. Submit the best supported "
                        "answer using final_answer."
                    ),
                    id="react:tool_limit",
                )
            )

        budget_message = HumanMessage(
            content=(
                f"Tool calls remaining: {tool_calls_remaining}. "
                f"Each call to {', '.join(tool.name for tool in budgeted_tools)} "
                "consumes one call."
            ),
            id=f"react:budget:{tool_call_count}",
        )
        invocation_messages = [*messages, *new_messages, budget_message]
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
                    format_error = "Python cannot be called after the execution limit"
                elif not isinstance(code, str) or not code.strip():
                    format_error = "Python tool call must contain non-empty code"
                elif purpose is not None and not isinstance(purpose, str):
                    format_error = "Python purpose must be text when provided"
                else:
                    history_entry["action"] = "python"
                    return {
                        **base_update,
                        "messages": [*new_messages, clean_message],
                        "agent_history": [*agent_history, history_entry],
                        "format_retry_count": 0,
                        "status": "tool_requested",
                    }

            elif cot_tool is not None and tool_name == cot_tool.name:
                goal = arguments.get("goal")
                if forced_final:
                    format_error = "CoT cannot be called after the tool limit"
                elif not isinstance(goal, str) or not goal.strip():
                    format_error = "CoT tool call must contain a non-empty goal"
                else:
                    history_entry["action"] = "cot"
                    return {
                        **base_update,
                        "messages": [*new_messages, clean_message],
                        "agent_history": [*agent_history, history_entry],
                        "format_retry_count": 0,
                        "status": "tool_requested",
                    }

            elif repair_tool is not None and tool_name == repair_tool.name:
                corrected_code = arguments.get("corrected_code")
                tool_history = list(state.get("tool_history", []))
                if forced_final:
                    format_error = "Repair cannot be called after the execution limit"
                elif not execution_failed(tool_history):
                    format_error = "Repair requires a previous failed execution"
                elif not isinstance(corrected_code, str) or not corrected_code.strip():
                    format_error = "Repair must contain non-empty corrected_code"
                else:
                    history_entry["action"] = "repair"
                    history_entry["repair_of_tool_call_id"] = tool_history[-1][
                        "tool_call_id"
                    ]
                    return {
                        **base_update,
                        "messages": [*new_messages, clean_message],
                        "agent_history": [*agent_history, history_entry],
                        "format_retry_count": 0,
                        "status": "tool_requested",
                    }

            elif tool_name == final_answer_tool.name:
                answer = arguments.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    format_error = "final_answer must contain a non-empty answer"
                else:
                    finish_reason = (
                        "tool_limit_reached"
                        if forced_final
                        else (
                            "format_recovery"
                            if state.get("had_format_recovery")
                            else "final_answer"
                        )
                    )
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
                *format_retry_messages(
                    message,
                    clean_message,
                    tool_calls_remaining,
                    available_tool_names,
                ),
            ],
            "agent_history": [*agent_history, history_entry],
            "format_retry_count": 1,
            "had_format_recovery": True,
            "status": "format_retry",
        }

    return agent


def route_after_react_agent(state: dict[str, Any]) -> str:
    """Направляет execution в ToolNode, format retry обратно в agent, ответ в END."""
    status = state["status"]
    if status == "tool_requested":
        return "tools"
    if status == "format_retry":
        return "agent"
    if status == "completed":
        return "finished"
    raise ValueError(f"Unknown ReAct agent status: {status}")


def record_react_tool_call(state: dict[str, Any]) -> dict[str, Any]:
    """Сохраняет artifact выполненного Python, repair или CoT tool."""
    messages = state["messages"]
    if len(messages) < 2:
        raise ValueError("Tool result has no matching agent message")

    agent_message = messages[-2]
    tool_message = messages[-1]
    if not isinstance(agent_message, AIMessage) or len(agent_message.tool_calls) != 1:
        raise ValueError("Tool result requires exactly one AI tool call")
    if not isinstance(tool_message, ToolMessage):
        raise ValueError("Tool execution did not return a ToolMessage")

    tool_call = agent_message.tool_calls[0]
    if tool_message.tool_call_id != tool_call["id"]:
        raise ValueError("ToolMessage does not match the requested tool call")
    if not isinstance(tool_message.artifact, dict):
        raise ValueError("ToolMessage has no artifact")

    arguments = tool_call["args"]
    tool_name = tool_call["name"]
    tool_index = state.get("tool_call_count", 0)
    tool_history = list(state.get("tool_history", []))
    state_update: dict[str, Any] = {}

    if tool_name == "python":
        purpose = arguments.get("purpose")
        purpose = purpose.strip() if isinstance(purpose, str) else None
        history_entry = {
            "index": tool_index,
            "tool_name": tool_name,
            "tool_call_id": tool_call["id"],
            "purpose": purpose,
            "code": arguments["code"].strip(),
            "execution": tool_message.artifact,
        }
    elif tool_name == "repair":
        history_entry = {
            "index": tool_index,
            "tool_name": tool_name,
            "tool_call_id": tool_call["id"],
            "purpose": None,
            "code": arguments["corrected_code"].strip(),
            "execution": tool_message.artifact,
            "repair_of_tool_call_id": state["tool_history"][-1]["tool_call_id"],
        }
    elif tool_name == "cot":
        artifact = tool_message.artifact
        cot_index = sum(item["tool_name"] == "cot" for item in tool_history)
        call_key = f"cot_{cot_index}"
        history_entry = {
            "index": tool_index,
            "tool_name": tool_name,
            "tool_call_id": tool_call["id"],
            "goal": arguments["goal"].strip(),
            "result": artifact.get("result", ""),
            "success": bool(artifact.get("success")),
            "latency_seconds": artifact.get("latency_seconds", 0.0),
            "usage": artifact.get("usage") or {},
        }
        usage = dict(state.get("usage", {}))
        usage[call_key] = artifact.get("usage") or {}
        state_update["usage"] = usage
        cot_reasoning = artifact.get("reasoning")
        if isinstance(cot_reasoning, str) and cot_reasoning:
            reasoning = dict(state.get("reasoning", {}))
            reasoning[call_key] = cot_reasoning
            state_update["reasoning"] = reasoning
    else:
        raise ValueError(f"Unsupported ReAct tool: {tool_name}")

    tool_history.append(history_entry)
    return {
        **state_update,
        "tool_call_count": tool_index + 1,
        "tool_history": tool_history,
        "status": "tool_executed",
    }
