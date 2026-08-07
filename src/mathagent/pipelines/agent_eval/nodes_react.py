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
from mathagent.tools.python_executor import NotebookSessionManager
from mathagent.tools.python_tools import (
    compact_execution,
    compact_stdout,
    execute_notebook_code,
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
    planner_trace: dict[str, Any] | None = None,
    repair_history: list[dict[str, Any]] | None = None,
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
    if planner_trace is not None:
        trace["planner"] = planner_trace
    if repair_history is not None:
        trace["repairs"] = repair_history
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


def create_react_planner_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Формирует один предварительный необязательный план перед ReAct-циклом."""
    prompt_version, role = load_prompt_role(prompt_path, "planner")

    def planner(state: dict[str, Any]) -> dict[str, Any]:
        task_prompt = role["task"].replace("{problem}", state["problem"])
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=role["system"]),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = time.perf_counter() - started
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("ReAct planner returned an empty plan")
        plan = message.content.strip()
        usage = get_message_usage(message)
        return {
            "plan": plan,
            "planner_trace": {
                "plan": plan,
                "usage": usage,
                "latency_seconds": round(latency, 3),
            },
            "reasoning": add_reasoning(state, "planner", message),
            "usage": add_usage(state, "planner", message),
            "prompt_version": prompt_version,
        }

    return planner


def create_react_agent_node(
    model: Any,
    execution_tools: list[BaseTool],
    final_answer_tool: BaseTool,
    prompt_path: Path,
    max_tool_calls: int,
    repair_tool: BaseTool | None = None,
    cot_tool: BaseTool | None = None,
    precheck_enabled: bool = False,
    planner_enabled: bool = False,
    structured_repair_enabled: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Вызывает ReAct-модель и обрабатывает tools выбранной версии промпта."""
    if repair_tool is not None and cot_tool is not None:
        raise ValueError("ReAct graph cannot enable cot and repair together")
    if not execution_tools:
        raise ValueError("ReAct graph requires at least one execution tool")

    execution_tools_by_name = {tool.name: tool for tool in execution_tools}
    if len(execution_tools_by_name) != len(execution_tools):
        raise ValueError("ReAct execution tool names must be unique")

    prompt_version, role = load_prompt_role(prompt_path, "agent")
    advisory_evidence = (
        "CoT, Python, or SymPy evidence"
        if "sympy" in execution_tools_by_name
        else "CoT or Python evidence"
    )
    budgeted_tools = list(execution_tools)
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
            if planner_enabled:
                plan = state.get("plan")
                if not isinstance(plan, str) or not plan.strip():
                    raise ValueError("ReAct agent requires a non-empty planner output")
                new_messages.append(
                    HumanMessage(
                        content=(
                            "Advisory initial plan from the planner:\n"
                            f"{plan}\n\n"
                            "Treat this plan as a hypothesis, not a fixed sequence. "
                            f"Revise or abandon any step when {advisory_evidence} "
                            "shows that another approach is better."
                        ),
                        id="react:advisory_plan",
                    )
                )

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

            if tool_name in execution_tools_by_name:
                tool_label = {
                    "python": "Python",
                    "sympy": "SymPy",
                }.get(str(tool_name), str(tool_name))
                code = arguments.get("code")
                purpose = arguments.get("purpose")
                context = arguments.get("context")
                if forced_final:
                    format_error = (
                        f"{tool_label} cannot be called after the execution limit"
                    )
                elif not isinstance(code, str) or not code.strip():
                    format_error = (
                        f"{tool_label} tool call must contain non-empty code"
                    )
                elif cot_tool is not None and (
                    not isinstance(context, str) or not context.strip()
                ):
                    format_error = (
                        f"{tool_label} tool call must contain non-empty context"
                    )
                elif purpose is not None and not isinstance(purpose, str):
                    format_error = (
                        f"{tool_label} purpose must be text when provided"
                    )
                else:
                    history_entry["action"] = tool_name
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
                            (
                                dict(state["planner_trace"])
                                if isinstance(state.get("planner_trace"), dict)
                                else None
                            ),
                            (
                                list(state.get("repair_history", []))
                                if structured_repair_enabled
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


def record_react_tool_call(
    state: dict[str, Any],
    *,
    structured_repair_enabled: bool = False,
    max_tool_repairs: int = 0,
) -> dict[str, Any]:
    """Сохраняет artifact выполненного execution, repair или CoT tool."""
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

    if tool_name in {"python", "sympy"}:
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
        if structured_repair_enabled:
            history_entry["recovered"] = False
            history_entry["repair_attempt_indices"] = []
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
    update = {
        **state_update,
        "tool_call_count": tool_index + 1,
        "tool_history": tool_history,
        "status": "tool_executed",
    }
    if (
        structured_repair_enabled
        and tool_name in {"python", "sympy"}
        and execution_failed(tool_history)
        and max_tool_repairs > 0
    ):
        update.update(
            {
                "repair_attempt": 0,
                "repair_code": history_entry["code"],
                "repair_execution": tool_message.artifact,
                "repair_tool_call_id": tool_call["id"],
                "repair_tool_name": tool_name,
                "repair_context": history_entry.get("context", ""),
                "repair_feedback": (
                    "No previous repair candidate has been rejected. Fix the latest "
                    "structured execution error."
                ),
                "status": "repair_required",
            }
        )
    return update


def route_after_react_execution(state: dict[str, Any]) -> str:
    """Выбирает автоматический repair после execution error или возврат агенту."""
    if state["status"] == "repair_required":
        return "repair"
    if state["status"] == "tool_executed":
        return "agent"
    raise ValueError(f"Unknown ReAct execution status: {state['status']}")


def repair_diagnostic(execution: dict[str, Any]) -> dict[str, Any]:
    """Выделяет structured diagnostic без полного stdout и traceback."""
    compact = compact_execution(execution)
    compact.pop("stdout", None)
    return compact


def normalize_repair_code(code: str) -> str:
    """Нормализует repair-код для исполнения и поиска повторных кандидатов."""
    value = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = value.splitlines()
    if lines and lines[0].strip().lower() in {"```", "```py", "```python"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(line.rstrip() for line in lines).strip()


def repair_diagnostic_excerpt(execution: dict[str, Any]) -> str:
    """Показывает repair-модели ошибочную строку и позицию compiler caret."""
    source_line = execution.get("source_line")
    line = execution.get("line")
    offset = execution.get("offset")
    if not isinstance(source_line, str) or not source_line:
        return "No exact source line is available; use the structured diagnostic."

    location = f"line {line}" if isinstance(line, int) else "unknown line"
    if isinstance(offset, int) and offset > 0:
        caret = " " * (offset - 1) + "^"
        return f"{location}:\n{source_line}\n{caret}"
    return f"{location}:\n{source_line}"


def repair_error_signature(execution: dict[str, Any]) -> tuple[Any, ...]:
    """Строит устойчивую сигнатуру ошибки для обнаружения повторного сбоя."""
    return (
        execution.get("error_type"),
        execution.get("message"),
        execution.get("line"),
        execution.get("source_line"),
        bool(execution.get("timeout")),
    )


def duplicate_repair_source(
    state: dict[str, Any],
    candidate: str,
) -> str | None:
    """Находит исходную или repair-ячейку, которую дословно повторил кандидат."""
    normalized_candidate = normalize_repair_code(candidate)
    tool_call_id = state["repair_tool_call_id"]
    for tool_call in reversed(state.get("tool_history", [])):
        if tool_call.get("tool_call_id") != tool_call_id:
            continue
        original_code = tool_call.get("code")
        if (
            isinstance(original_code, str)
            and normalize_repair_code(original_code) == normalized_candidate
        ):
            return "original"
        break

    for repair in state.get("repair_history", []):
        if repair.get("tool_call_id") != tool_call_id:
            continue
        previous_code = repair.get("code")
        if (
            isinstance(previous_code, str)
            and normalize_repair_code(previous_code) == normalized_candidate
        ):
            return f"repair_{repair['index']}"
    return None


def duplicate_repair_execution(
    previous_execution: dict[str, Any],
    duplicate_of: str,
) -> dict[str, Any]:
    """Создаёт trace-диагностику для кандидата, не запущенного как дубликат."""
    return {
        "executed": False,
        "success": False,
        "stdout": "",
        "stderr": "",
        "error_type": "DuplicateRepairCandidate",
        "message": f"Repair candidate duplicates {duplicate_of}",
        "line": previous_execution.get("line"),
        "offset": previous_execution.get("offset"),
        "source_line": previous_execution.get("source_line"),
        "returncode": None,
        "timeout": False,
        "latency_seconds": 0.0,
        "session_reset": False,
        "session_version": previous_execution.get("session_version"),
        "available_names": list(previous_execution.get("available_names", [])),
        "defined_names": [],
        "names_truncated": bool(previous_execution.get("names_truncated", False)),
    }


def next_repair_feedback(
    previous_execution: dict[str, Any],
    execution: dict[str, Any],
) -> str:
    """Объясняет следующей repair-попытке, чем закончился новый кандидат."""
    error_type = execution.get("error_type") or "execution error"
    if repair_error_signature(previous_execution) == repair_error_signature(execution):
        source_line = execution.get("source_line") or "unknown source line"
        return (
            "The previous candidate changed, but it produced the same "
            f"{error_type} with the same diagnostic. Rewrite the complete failing "
            f"statement instead of preserving its token pattern: {source_line}"
        )
    return (
        "The previous candidate was executed but failed with "
        f"{error_type}: {execution.get('message') or 'no error message'}. "
        "Use the latest diagnostic and return a different, corrected cell."
    )


def repair_exhaustion_content(
    execution: dict[str, Any],
    repair_history: list[dict[str, Any]],
    tool_call_id: str,
) -> str:
    """Собирает последнюю реальную ошибку и сводку исчерпанного repair-loop."""
    current_repairs = [
        repair
        for repair in repair_history
        if repair.get("tool_call_id") == tool_call_id
    ]
    duplicate_count = sum(
        repair.get("candidate_validation", {}).get("reason") == "duplicate_code"
        for repair in current_repairs
    )
    payload = {
        "last_execution_error": repair_diagnostic(execution),
        "repair_attempts": len(current_repairs),
        "duplicate_candidates_rejected": duplicate_count,
    }
    return json.dumps(payload, ensure_ascii=False)


def create_react_code_repair_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Генерирует полную исправленную ячейку по последней execution-ошибке."""
    prompt_version, role = load_prompt_role(prompt_path, "repair")
    candidate_validation_enabled = "{repair_feedback}" in role["task"]

    def repair(state: dict[str, Any]) -> dict[str, Any]:
        attempt = state.get("repair_attempt", 0) + 1
        execution = state["repair_execution"]
        replacements = {
            "problem": state["problem"],
            "tool_name": state["repair_tool_name"],
            "context": state.get("repair_context", ""),
            "code": state["repair_code"],
            "diagnostic": json.dumps(
                repair_diagnostic(execution),
                ensure_ascii=False,
                indent=2,
            ),
            "diagnostic_excerpt": repair_diagnostic_excerpt(execution),
            "repair_feedback": state.get(
                "repair_feedback",
                "No previous repair candidate has been rejected.",
            ),
            "stdout": compact_stdout(execution.get("stdout", "")),
            "available_names": json.dumps(
                execution.get("available_names", []),
                ensure_ascii=False,
            ),
            "repair_attempt": str(attempt),
        }
        task_prompt = role["task"]
        for field, value in replacements.items():
            task_prompt = task_prompt.replace(f"{{{field}}}", value)

        repair_index = len(state.get("repair_history", []))
        call_key = f"repair_{repair_index}"
        started = time.perf_counter()
        message = model.invoke(
            [
                SystemMessage(content=role["system"]),
                HumanMessage(content=task_prompt),
            ]
        )
        latency = time.perf_counter() - started
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("ReAct code repair returned empty code")
        repair_code = (
            normalize_repair_code(message.content)
            if candidate_validation_enabled
            else message.content.strip()
        )
        if not repair_code:
            raise ValueError("ReAct code repair returned empty code")
        update = {
            "repair_code": repair_code,
            "repair_attempt": attempt,
            "repair_model_usage": get_message_usage(message),
            "repair_model_latency": round(latency, 3),
            "reasoning": add_reasoning(state, call_key, message),
            "usage": add_usage(state, call_key, message),
            "prompt_version": prompt_version,
            "status": "repair_generated",
        }
        if candidate_validation_enabled:
            duplicate_of = duplicate_repair_source(state, repair_code)
            update["repair_candidate_validation"] = {
                "accepted": duplicate_of is None,
                "reason": "duplicate_code" if duplicate_of is not None else None,
                "duplicate_of": duplicate_of,
            }
        return update

    return repair


def create_react_repair_executor_node(
    manager: NotebookSessionManager,
    max_tool_repairs: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Исполняет repair-ячейку и завершает либо продолжает локальный цикл."""

    def execute_repair(state: dict[str, Any]) -> dict[str, Any]:
        session_id = state.get("python_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("ReAct code repair requires a notebook session id")

        previous_execution = state["repair_execution"]
        candidate_validation_value = state.get("repair_candidate_validation")
        candidate_validation = (
            dict(candidate_validation_value)
            if isinstance(candidate_validation_value, dict)
            else None
        )
        if candidate_validation is None or candidate_validation["accepted"]:
            content, raw_execution = execute_notebook_code(
                manager,
                session_id,
                state["repair_code"],
            )
            execution = (
                {**raw_execution, "executed": True}
                if candidate_validation is not None
                else raw_execution
            )
        else:
            duplicate_of = candidate_validation["duplicate_of"]
            execution = duplicate_repair_execution(
                previous_execution,
                duplicate_of,
            )
            content = json.dumps(compact_execution(execution), ensure_ascii=False)
        repair_history = list(state.get("repair_history", []))
        repair_index = len(repair_history)
        repair_entry = {
            "index": repair_index,
            "tool_call_id": state["repair_tool_call_id"],
            "tool_name": state["repair_tool_name"],
            "attempt": state["repair_attempt"],
            "code": state["repair_code"],
            "execution": execution,
            "usage": state.get("repair_model_usage", {}),
            "latency_seconds": state.get("repair_model_latency", 0.0),
        }
        if candidate_validation is not None:
            repair_entry["candidate_validation"] = candidate_validation
        repair_history.append(repair_entry)

        tool_history = list(state["tool_history"])
        origin = dict(tool_history[-1])
        attempt_indices = list(origin.get("repair_attempt_indices", []))
        attempt_indices.append(repair_index)
        origin["repair_attempt_indices"] = attempt_indices
        succeeded = execution.get("returncode") == 0 and not execution.get("timeout")
        origin["recovered"] = succeeded
        tool_history[-1] = origin

        real_execution = (
            execution
            if candidate_validation is None or candidate_validation["accepted"]
            else previous_execution
        )
        update: dict[str, Any] = {
            "repair_execution": real_execution,
            "repair_history": repair_history,
            "tool_history": tool_history,
        }
        if succeeded:
            update.update(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "The automatic code repair succeeded. Its final "
                                f"execution result is:\n{content}"
                            ),
                            id=f"react:repair_result:{repair_index}",
                        )
                    ],
                    "status": "repair_succeeded",
                }
            )
        elif state["repair_attempt"] >= max_tool_repairs:
            exhaustion = (
                repair_exhaustion_content(
                    real_execution,
                    repair_history,
                    state["repair_tool_call_id"],
                )
                if candidate_validation is not None
                else content
            )
            update.update(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "Automatic code repair exhausted its retry limit. "
                                "Use this final structured error to choose a new "
                                f"action:\n{exhaustion}"
                            ),
                            id=f"react:repair_exhausted:{repair_index}",
                        )
                    ],
                    "status": "repair_exhausted",
                }
            )
        else:
            if candidate_validation is None:
                feedback = None
            elif candidate_validation["accepted"]:
                feedback = next_repair_feedback(previous_execution, execution)
            else:
                feedback = (
                    "The previous candidate was rejected without execution because "
                    f"it duplicated {candidate_validation['duplicate_of']}. Return "
                    "different code that rewrites the exact failing statement."
                )
            if feedback is not None:
                update["repair_feedback"] = feedback
            update["status"] = "repair_failed"
        return update

    return execute_repair


def route_after_react_repair(state: dict[str, Any]) -> str:
    """Повторяет repair после ошибки либо возвращает управление агенту."""
    if state["status"] == "repair_failed":
        return "repair"
    if state["status"] in {"repair_succeeded", "repair_exhausted"}:
        return "agent"
    raise ValueError(f"Unknown ReAct repair status: {state['status']}")
