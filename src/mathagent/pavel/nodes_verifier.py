import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mathagent.pavel.nodes import add_reasoning, add_usage, get_message_usage
from mathagent.pavel.nodes_react import build_react_trace


LOGGER = logging.getLogger(__name__)

_NEGATIVE_CHECK_MARKERS = [
    "ошибк",
    "error",
    "неверно",
    "некоррект",
    "несовпада",
    "не совпада",
]
_POSITIVE_CHECK_MARKERS = ["ok", "верно", "корректно", "правильно"]


def _parse_check_status(check_raw: str, logger: Any = LOGGER) -> bool:
    """Возвращает True (OK) / False (ERROR) на основе текста тега <check>"""
    normalized = (check_raw or "").strip().lower()

    for marker in _NEGATIVE_CHECK_MARKERS:
        if marker in normalized:
            return False

    for marker in _POSITIVE_CHECK_MARKERS:
        if marker in normalized:
            return True

    logger.warning(
        "Не удалось распознать статус шага из значения <check>: '%s'. "
        "По умолчанию считаю шаг ERROR",
        check_raw,
    )
    return False


def _get_done_reason(message: Any) -> str | None:
    """Достаёт причину завершения из OpenAI-compatible ответа модели"""
    return get_message_usage(message).get("finish_reason")


def _extract_tag(tag_name: str, source_text: str) -> str | None:
    """Извлекает содержимое между указанными тегами"""
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, source_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


_STEP_PATTERN = re.compile(
    r"<text>(.*?)</text>\s*"
    r"(?:<scratch>(.*?)</scratch>\s*)?"
    r"<check>(.*?)</check>\s*"
    r"(?:<comment>(.*?)</comment>)?",
    re.DOTALL | re.IGNORECASE,
)


def _extract_steps(source_text: str, logger: Any = LOGGER) -> list[dict[str, Any]]:
    """Извлекает список пошаговых проверок из блока <steps>"""
    steps_block = _extract_tag("steps", source_text)
    if not steps_block:
        logger.warning("Блок <steps> не найден в ответе модели")
        return []

    matches = list(_STEP_PATTERN.finditer(steps_block))
    parsed_steps: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for index, match in enumerate(matches, start=1):
        text = (match.group(1) or "").strip() or "Модель не описала шаг."
        scratch = (match.group(2) or "").strip()
        check_raw = (match.group(3) or "").strip()
        comment = (match.group(4) or "").strip()

        normalized = text.lower()
        if normalized in seen_texts:
            logger.warning(
                "Обнаружено зацикливание verifier на шаге %d: %s",
                index,
                text[:60],
            )
            break
        seen_texts.add(normalized)
        is_ok = _parse_check_status(check_raw, logger)
        if scratch:
            logger.info("verifier_step=%d scratch=%s", index, scratch)
        parsed_steps.append(
            {
                "step_number": index,
                "text": text,
                "check": "OK" if is_ok else "ERROR",
                "is_ok": is_ok,
                "comment": comment,
            }
        )

    if not parsed_steps:
        logger.warning("Блок <steps> найден, но ни один шаг не распарсился")
    return parsed_steps


def _render_steps_block(steps: list[dict[str, Any]], was_truncated: bool) -> str:
    """Превращает распарсенные шаги в текст для второго вызова verifier"""
    if not steps:
        note = " (генерация была обрезана по лимиту токенов)" if was_truncated else ""
        return f"Шаги не получены{note}. Проверка неполная."

    lines = [
        (
            f"Шаг {step['step_number']}: {step['text']} — {step['check']}. "
            f"{step['comment']}"
        )
        for step in steps
    ]
    if was_truncated:
        lines.append(
            "[Внимание: разбор по шагам оборвался по лимиту токенов раньше, "
            "чем решение студента было проверено полностью — выше приведены "
            "только полностью проверенные шаги.]"
        )
    return "\n".join(lines)


def _clean_and_parse_finalize(
    raw_text: str,
    done_reason: str | None,
    *,
    prefill: str,
    steps_present: bool,
    steps_truncated: bool,
    logger: Any = LOGGER,
) -> dict[str, Any]:
    """Разбирает итоговые <verdict>, <feedback> и <analysis>"""
    text = raw_text.strip()
    if not text.startswith("<verdict>"):
        text = prefill + text

    was_truncated = done_reason == "length"
    if was_truncated:
        logger.warning("Финальный verifier оборвался по лимиту токенов")

    defaults = {
        "verdict": "<verdict>FALSE</verdict>",
        "feedback": (
            "<feedback>Ответ модели-верификатора не содержал этот блок.</feedback>"
        ),
        "analysis": (
            "<analysis>Ответ модели-верификатора не содержал этот блок.</analysis>"
        ),
    }
    for tag in ("verdict", "feedback", "analysis"):
        open_tag, close_tag = f"<{tag}>", f"</{tag}>"
        if open_tag in text and close_tag not in text:
            text = text.rstrip() + f"\n{close_tag}"
        elif open_tag not in text:
            if done_reason != "length":
                was_truncated = True
            text = text.rstrip() + "\n" + defaults[tag]

    verdict_raw = _extract_tag("verdict", text) or ""
    analysis = _extract_tag("analysis", text) or "Модель не заполнила описание."
    feedback = _extract_tag("feedback", text) or "Решение требует проверки."
    is_correct = "true" in verdict_raw.strip().lower()

    if not steps_present and is_correct:
        is_correct = False
        analysis = (
            analysis
            + " [Внимание: пошаговая проверка не дала ни одного шага, "
            "вердикт TRUE не подтверждён.]"
        ).strip()

    return {
        "is_correct_solution": is_correct,
        "analysis_thoughts": analysis,
        "feedback_for_solver": feedback,
        "is_truncated": was_truncated or steps_truncated,
    }


def _load_verifier_role(prompt_path: Path, role_name: str) -> dict[str, Any]:
    """Загружает роль из оригинального prompt-конфига verifier"""
    with prompt_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    try:
        role = config[role_name]
        system_prompt = role["system_prompt"]
        user_template = role["user_prompt_template"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid verifier role '{role_name}': {prompt_path}"
        ) from exc
    if not isinstance(system_prompt, str) or not isinstance(user_template, str):
        raise ValueError(f"Invalid verifier role '{role_name}': {prompt_path}")
    return {
        "system_prompt": system_prompt,
        "user_prompt_template": user_template,
        "prefill": str(role.get("prefill", "") or ""),
        "stop": list(role.get("stop", []) or []),
    }


def _build_messages(role: dict[str, Any], **values: str) -> list[Any]:
    """Собирает system, user и исходный assistant-prefill как в исходной ветке"""
    messages: list[Any] = [
        SystemMessage(content=role["system_prompt"]),
        HumanMessage(content=role["user_prompt_template"].format(**values)),
    ]
    if role["prefill"]:
        messages.append(AIMessage(content=role["prefill"]))
    return messages


def _run_stepwise_check(
    model: Any,
    role: dict[str, Any],
    question: str,
    model_answer: str,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any], Any]:
    """Вызывает первый этап исходного verifier и парсит OK/ERROR шаги"""
    started = time.perf_counter()
    message = model.invoke(
        _build_messages(role, question=question, model_answer=model_answer),
        stop=role["stop"],
    )
    latency = round(time.perf_counter() - started, 3)
    raw_content = message.content if isinstance(message.content, str) else ""
    done_reason = _get_done_reason(message)
    text = raw_content.strip()

    if not text.startswith("<steps>"):
        text = role["prefill"] + text

    was_truncated = done_reason == "length"
    if "</steps>" not in text:
        text = text.rstrip() + "\n</steps>"
        if done_reason not in ("length", "stop"):
            was_truncated = True

    steps = _extract_steps(text)
    trace = {
        "raw_content": raw_content,
        "done_reason": done_reason,
        "was_truncated": was_truncated,
        "usage": get_message_usage(message),
        "latency_seconds": latency,
    }
    return steps, was_truncated, trace, message


def _run_finalize(
    model: Any,
    role: dict[str, Any],
    question: str,
    model_answer: str,
    steps_block: str,
    *,
    steps_present: bool,
    steps_truncated: bool,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Вызывает второй этап исходного verifier и парсит TRUE/FALSE"""
    started = time.perf_counter()
    message = model.invoke(
        _build_messages(
            role,
            question=question,
            model_answer=model_answer,
            steps_block=steps_block,
        ),
        stop=role["stop"],
    )
    latency = round(time.perf_counter() - started, 3)
    raw_content = message.content if isinstance(message.content, str) else ""
    done_reason = _get_done_reason(message)
    result = _clean_and_parse_finalize(
        raw_content,
        done_reason,
        prefill=role["prefill"],
        steps_present=steps_present,
        steps_truncated=steps_truncated,
    )
    trace = {
        "raw_content": raw_content,
        "done_reason": done_reason,
        "usage": get_message_usage(message),
        "latency_seconds": latency,
    }
    return result, trace, message


def create_react_verifier_node(
    stepwise_model: Any,
    finalize_model: Any,
    agent_prompt_path: Path,
    verifier_prompt_path: Path,
    stepwise_role_name: str,
    finalize_role_name: str,
    max_verification_rounds: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Адаптирует двухэтапный verifier из ветки Стаса к ReAct-loop"""
    with agent_prompt_path.open(encoding="utf-8") as stream:
        agent_prompt = yaml.safe_load(stream)
    prompt_version = str(agent_prompt["version"])
    stepwise_role = _load_verifier_role(verifier_prompt_path, stepwise_role_name)
    finalize_role = _load_verifier_role(verifier_prompt_path, finalize_role_name)

    def verifier(state: dict[str, Any]) -> dict[str, Any]:
        round_index = state.get("verification_round", 0)
        answer = state.get("proposed_answer")
        solution = state.get("proposed_solution")
        tool_call_id = state.get("proposed_tool_call_id")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Verifier requires a non-empty proposed answer")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("Verifier requires a non-empty proposed solution")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("Verifier requires the final_solution tool call id")

        steps, steps_truncated, stepwise_trace, stepwise_message = (
            _run_stepwise_check(
                stepwise_model,
                stepwise_role,
                state["problem"],
                solution,
            )
        )
        result, finalize_trace, finalize_message = _run_finalize(
            finalize_model,
            finalize_role,
            state["problem"],
            solution,
            _render_steps_block(steps, steps_truncated),
            steps_present=bool(steps),
            steps_truncated=steps_truncated,
        )
        result["steps"] = steps
        verification = {
            **result,
            "stepwise": stepwise_trace,
            "finalize": finalize_trace,
        }

        attempts = list(state.get("solution_attempts", []))
        attempts.append(
            {
                "index": round_index,
                "answer": answer.strip(),
                "solution": solution.strip(),
                "verification": verification,
            }
        )
        next_round = round_index + 1
        stepwise_key = f"verifier_stepwise_{round_index}"
        finalize_key = f"verifier_finalize_{round_index}"
        state_with_stepwise = {
            **state,
            "reasoning": add_reasoning(state, stepwise_key, stepwise_message),
            "usage": add_usage(state, stepwise_key, stepwise_message),
        }
        reasoning = add_reasoning(
            state_with_stepwise,
            finalize_key,
            finalize_message,
        )
        usage = add_usage(state_with_stepwise, finalize_key, finalize_message)
        failed_steps = [step for step in steps if not step["is_ok"]]
        feedback_payload = {
            "is_correct_solution": result["is_correct_solution"],
            "feedback_for_solver": result["feedback_for_solver"],
            "analysis_thoughts": result["analysis_thoughts"],
            "failed_steps": failed_steps,
            "is_truncated": result["is_truncated"],
        }
        if not result["is_correct_solution"]:
            feedback_payload.update(
                {
                    "rejected_answer": answer.strip(),
                    "rejected_solution": solution.strip(),
                }
            )
        base_update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    content=json.dumps(feedback_payload, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                    name="final_solution",
                    id=f"react:verification:{round_index}",
                )
            ],
            "solution_attempts": attempts,
            "verification_round": next_round,
            "reasoning": reasoning,
            "usage": usage,
            "prompt_version": prompt_version,
        }

        if result["is_correct_solution"]:
            finish_reason = "verified"
        elif next_round >= max_verification_rounds:
            finish_reason = "verification_limit_reached"
        else:
            return {**base_update, "status": "verification_feedback"}

        return {
            **base_update,
            "solution": solution.strip(),
            "status": finish_reason,
            "finish_reason": finish_reason,
            "trace": build_react_trace(
                list(state.get("agent_history", [])),
                list(state.get("tool_history", [])),
                finish_reason,
                (
                    list(state.get("precheck_history", []))
                    if "precheck_history" in state
                    else None
                ),
                (
                    dict(state["planner_trace"])
                    if isinstance(state.get("planner_trace"), dict)
                    else None
                ),
                (
                    list(state.get("repair_history", []))
                    if "repair_history" in state
                    else None
                ),
                solution_attempts=attempts,
                selected_attempt=len(attempts) - 1,
            ),
        }

    return verifier


def route_after_react_verification(state: dict[str, Any]) -> str:
    """Возвращает feedback агенту или завершает проверенный pipeline"""
    if state["status"] == "verification_feedback":
        return "agent"
    if state["status"] in {"verified", "verification_limit_reached"}:
        return "finished"
    raise ValueError(f"Unknown ReAct verification status: {state['status']}")
