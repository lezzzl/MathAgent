import ast
import io
import json
import time
import tokenize
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
    precheck_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Собирает компактную траекторию без messages и hidden reasoning."""
    trace = {
        "status": "completed",
        "finish_reason": finish_reason,
        "agent_calls": agent_history,
        "tool_calls": tool_history,
    }
    if precheck_history is not None:
        trace["prechecks"] = precheck_history
    return trace


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
    precheck_enabled: bool = False,
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
        force_final_reason = state.get("force_final_reason")
        forced_final = tool_calls_remaining == 0 or force_final_reason is not None
        if forced_final:
            reason = (
                "The precheck rejection limit has been reached"
                if force_final_reason == "precheck_limit_reached"
                else "The tool-call limit has been reached"
            )
            new_messages.append(
                HumanMessage(
                    content=(
                        f"{reason}. Do not call any "
                        "reasoning or execution tool. Submit the best supported "
                        "answer using final_answer."
                    ),
                    id="react:forced_final",
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
                context = arguments.get("context")
                if forced_final:
                    format_error = "Python cannot be called after the execution limit"
                elif not isinstance(code, str) or not code.strip():
                    format_error = "Python tool call must contain non-empty code"
                elif cot_tool is not None and (
                    not isinstance(context, str) or not context.strip()
                ):
                    format_error = "Python tool call must contain non-empty context"
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
                context = arguments.get("context")
                if forced_final:
                    format_error = "CoT cannot be called after the tool limit"
                elif not isinstance(goal, str) or not goal.strip():
                    format_error = "CoT tool call must contain a non-empty goal"
                elif not isinstance(context, str) or not context.strip():
                    format_error = "CoT tool call must contain non-empty context"
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
                        str(force_final_reason)
                        if force_final_reason is not None
                        else "tool_limit_reached"
                        if tool_calls_remaining == 0
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
                            (
                                list(state.get("precheck_history", []))
                                if precheck_enabled
                                else None
                            ),
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
                    0 if forced_final else tool_calls_remaining,
                    available_tool_names,
                ),
            ],
            "agent_history": [*agent_history, history_entry],
            "format_retry_count": 1,
            "had_format_recovery": True,
            "status": "format_retry",
        }

    return agent


def python_precheck_error(code: str) -> str | None:
    """Проверяет синтаксис Python и наличие настоящего комментария."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        location = f" at line {exc.lineno}" if exc.lineno is not None else ""
        return f"Python code has a syntax error{location}: {exc.msg}"

    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        comments = [
            token.string.removeprefix("#").strip()
            for token in tokens
            if token.type == tokenize.COMMENT
        ]
    except (IndentationError, tokenize.TokenError) as exc:
        return f"Python code cannot be tokenized: {exc}"
    if not any(comments):
        return (
            "Python code must include Program-of-Thought comments explaining "
            "the mathematical reasoning and purpose of the computation"
        )
    return None


def create_react_precheck_node(
    model: Any,
    review_tool: BaseTool,
    prompt_path: Path,
    max_precheck_rejections: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Проверяет предложенный CoT/Python-вызов до его исполнения."""
    _, role = load_prompt_role(prompt_path, "tool_checker")
    model_with_review = model.bind_tools(
        [review_tool],
        tool_choice=review_tool.name,
        strict=True,
        parallel_tool_calls=False,
    )

    def precheck(state: dict[str, Any]) -> dict[str, Any]:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or len(message.tool_calls) != 1:
            raise ValueError("Precheck requires exactly one proposed tool call")

        tool_call = message.tool_calls[0]
        tool_name = tool_call["name"]
        arguments = tool_call.get("args") or {}
        if tool_name not in {"python", "cot"}:
            raise ValueError(f"Precheck does not support tool: {tool_name}")

        precheck_history = list(state.get("precheck_history", []))
        index = len(precheck_history)
        call_key = f"precheck_{index}"
        decision = "reject"
        feedback: str
        source = "rules"
        latency = 0.0
        usage: dict[str, Any] = {}
        state_update: dict[str, Any] = {}

        rule_error = None
        if tool_name == "python":
            code = arguments.get("code")
            if not isinstance(code, str):
                rule_error = "Python code must be text"
            else:
                rule_error = python_precheck_error(code)

        if rule_error is not None:
            feedback = rule_error
        else:
            task_prompt = (
                role["task"]
                .replace("{problem}", state["problem"])
                .replace("{tool_name}", tool_name)
                .replace(
                    "{tool_arguments}",
                    json.dumps(arguments, ensure_ascii=False, indent=2),
                )
            )
            started = time.perf_counter()
            review_message = model_with_review.invoke(
                [
                    SystemMessage(content=role["system"]),
                    HumanMessage(content=task_prompt),
                ]
            )
            latency = time.perf_counter() - started
            source = "llm"
            usage = get_message_usage(review_message)
            state_update = {
                "reasoning": add_reasoning(state, call_key, review_message),
                "usage": add_usage(state, call_key, review_message),
            }
            if review_message.invalid_tool_calls or len(review_message.tool_calls) != 1:
                feedback = (
                    "The tool checker returned an invalid verdict. Regenerate the "
                    "proposed tool call before execution"
                )
            else:
                verdict = review_message.tool_calls[0]
                verdict_arguments = verdict.get("args") or {}
                verdict_decision = verdict_arguments.get("decision")
                verdict_feedback = verdict_arguments.get("feedback")
                if (
                    verdict.get("name") == review_tool.name
                    and verdict_decision in {"approve", "reject"}
                    and isinstance(verdict_feedback, str)
                    and verdict_feedback.strip()
                ):
                    decision = verdict_decision
                    feedback = verdict_feedback.strip()
                else:
                    feedback = (
                        "The tool checker returned a malformed verdict. Regenerate "
                        "the proposed tool call before execution"
                    )

        history_entry: dict[str, Any] = {
            "index": index,
            "tool_call_id": tool_call["id"],
            "tool_name": tool_name,
            "decision": decision,
            "source": source,
            "feedback": feedback,
            "latency_seconds": round(latency, 3),
            "usage": usage,
        }
        if decision == "approve":
            return {
                **state_update,
                "precheck_history": [*precheck_history, history_entry],
                "status": "tool_approved",
            }

        history_entry["arguments"] = arguments
        rejection_count = state.get("precheck_rejection_count", 0) + 1
        rejection_message = ToolMessage(
            content=json.dumps(
                {
                    "status": "rejected_by_precheck",
                    "feedback": feedback,
                },
                ensure_ascii=False,
            ),
            tool_call_id=tool_call["id"],
            name=tool_name,
            status="error",
            id=f"react:precheck_rejected:{tool_call['id']}",
        )
        return {
            **state_update,
            "messages": [rejection_message],
            "precheck_history": [*precheck_history, history_entry],
            "precheck_rejection_count": rejection_count,
            **(
                {"force_final_reason": "precheck_limit_reached"}
                if rejection_count >= max_precheck_rejections
                else {}
            ),
            "status": "tool_rejected",
        }

    return precheck


def route_after_react_precheck(state: dict[str, Any]) -> str:
    """Исполняет одобренный tool либо возвращает отклонение агенту."""
    if state["status"] == "tool_approved":
        return "tools"
    if state["status"] == "tool_rejected":
        return "agent"
    raise ValueError(f"Unknown ReAct precheck status: {state['status']}")


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
        context = arguments.get("context")
        context = context.strip() if isinstance(context, str) else None
        history_entry = {
            "index": tool_index,
            "tool_name": tool_name,
            "tool_call_id": tool_call["id"],
            "code": arguments["code"].strip(),
            "execution": tool_message.artifact,
        }
        if context is not None:
            history_entry["context"] = context
        else:
            history_entry["purpose"] = purpose
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
            "context": arguments["context"].strip(),
            "derivation": artifact.get("derivation", ""),
            "result": artifact.get("result", ""),
            "success": bool(artifact.get("success")),
            "parse_mode": artifact.get("parse_mode"),
            "parse_warning": artifact.get("parse_warning"),
            "latency_seconds": artifact.get("latency_seconds", 0.0),
            "usage": artifact.get("usage") or {},
        }
        if artifact.get("parse_mode") != "json":
            history_entry["raw_output"] = artifact.get("raw_output", "")
        if artifact.get("error") is not None:
            history_entry["error"] = artifact["error"]
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
