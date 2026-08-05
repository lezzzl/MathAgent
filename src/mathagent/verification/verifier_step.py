import re
from pathlib import Path

import ollama

from roles import Role

MODEL_NAME = 'qwen3.5:9b'
PROMPTS_PATH = Path("prompts/verificator-stepwise.yml")

_stepwise_role = Role.from_yaml(PROMPTS_PATH, "verifier_stepwise_check")
_finalize_role = Role.from_yaml(PROMPTS_PATH, "verifier_finalize")

_NEGATIVE_CHECK_MARKERS = ["ошибк", "error", "неверно", "некоррект", "несовпада", "не совпада"]
_POSITIVE_CHECK_MARKERS = ["ok", "верно", "корректно", "правильно"]


def _parse_check_status(check_raw: str, logger) -> bool:
    """Возвращает True (OK) / False (ERROR) на основе текста тега <check>."""
    normalized = (check_raw or "").strip().lower()

    for marker in _NEGATIVE_CHECK_MARKERS:
        if marker in normalized:
            return False

    for marker in _POSITIVE_CHECK_MARKERS:
        if marker in normalized:
            return True

    logger.warning(
        f"Не удалось распознать статус шага из значения <check>: '{check_raw}'. "
        "По умолчанию считаю шаг ERROR (неоднозначный сигнал не должен молча "
        "засчитываться как подтверждение верности)."
    )
    return False


def _get_done_reason(response):
    """
    Достаёт done_reason из ответа Ollama ('stop' — дошли до стоп-
    последовательности штатно, 'length' — упёрлись в num_predict).
    """
    try:
        value = response.get('done_reason')
        if value is not None:
            return value
    except AttributeError:
        pass
    return getattr(response, 'done_reason', None)


def _extract_tag(tag_name: str, source_text: str):
    """Извлекает содержимое между указанными тегами (первое вхождение)."""
    pattern = fr"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, source_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


_STEP_PATTERN = re.compile(
    r"<text>(.*?)</text>\s*"
    r"(?:<scratch>(.*?)</scratch>\s*)?"
    r"<check>(.*?)</check>\s*"
    r"(?:<comment>(.*?)</comment>)?",
    re.DOTALL | re.IGNORECASE,
)


def _extract_steps(source_text: str, logger):
    """
    извлекает список пошаговых проверок из блока <steps>...</steps>.
    """
    steps_block = _extract_tag("steps", source_text)
    if not steps_block:
        logger.warning("Блок <steps> не найден в ответе модели (шаг 1).")
        return []

    matches = list(_STEP_PATTERN.finditer(steps_block))
    parsed_steps = []
    seen_texts = set()

    for idx, m in enumerate(matches, start=1):
        text = (m.group(1) or "").strip() or "Модель не описала шаг."
        scratch = (m.group(2) or "").strip()
        check_raw = (m.group(3) or "").strip()
        comment = (m.group(4) or "").strip()

        normalized = text.lower()
        if normalized in seen_texts:
            logger.warning(
                f"Обнаружено зацикливание генерации на шаге {idx} "
                f"(повтор текста шага: '{text[:60]}...'). "
                f"Отбрасываю этот и все последующие шаги."
            )
            break
        seen_texts.add(normalized)
        is_ok = _parse_check_status(check_raw, logger)
        if scratch:
            logger.info(f"step {idx} scratch: {scratch}")
        parsed_steps.append({
            "step_number": idx,
            "text": text,
            "check": "OK" if is_ok else "ERROR",
            "is_ok": is_ok,
            "comment": comment,
        })

    if not parsed_steps:
        logger.warning("Блок <steps> присутствует, но ни один шаг не распарсился.")

    return parsed_steps


def _run_stepwise_check(question: str, model_answer: str, logger):
    """
    Получить от LLM разбор решения студента на шаги (OK/ERROR).
    """
    messages = _stepwise_role.build_messages(question=question, model_answer=model_answer)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                'temperature': 0.1,
                'num_ctx': 16384,
                'num_predict': 10000,
                'stop': _stepwise_role.stop,
            }
        )
    except Exception as e:
        logger.error(f"Сбой при подключении к LLM на шаге разбора по шагам: {e}")
        return [], True

    raw_content = response['message'].get('content', '')
    done_reason = _get_done_reason(response)
    logger.info(f"stepwise_check_raw_content: {raw_content}")
    logger.info(f"stepwise_check_done_reason: {done_reason}")

    text = raw_content.strip()

    if not text.startswith("<steps>"):
        text = _stepwise_role.prefill + text

    was_truncated = done_reason == 'length'
    if was_truncated:
        logger.warning(
            "Генерация оборвалась по num_predict на шаге разбора по шагам (шаг 1). "
            "Учтены только полностью сгенерированные шаги."
        )

    if "</steps>" not in text:
        # Штатный случай: done_reason == 'stop' => тег вырезан стоп-
        # последовательностью, просто дописываем его обратно.
        # done_reason == 'length' => реальный обрыв, тоже закрываем, чтобы
        # не терять уже сгенерированные (полностью закрытые) шаги.
        text = text.rstrip() + "\n</steps>"
        if done_reason not in ('length', 'stop'):
            was_truncated = True
            logger.warning(
                f"Неожиданный done_reason='{done_reason}' без </steps> в ответе — "
                "на всякий случай помечаю как truncated."
            )

    steps = _extract_steps(text, logger)
    return steps, was_truncated


def _render_steps_block(steps: list, was_truncated: bool) -> str:
    if not steps:
        note = " (генерация была обрезана по лимиту токенов)" if was_truncated else ""
        return f"Шаги не получены{note}. Проверка неполная."

    lines = []
    for step in steps:
        lines.append(
            f"Шаг {step['step_number']}: {step['text']} — {step['check']}. {step['comment']}"
        )
    if was_truncated:
        lines.append(
            "[Внимание: разбор по шагам оборвался по лимиту токенов раньше, чем решение "
            "студента было проверено полностью — выше приведены только полностью "
            "проверенные шаги.]"
        )
    return "\n".join(lines)


def _run_finalize(question: str, model_answer: str, steps_block: str, logger):
    """Агрегировать готовый разбор шагов в вердикт/фидбек."""
    messages = _finalize_role.build_messages(
        question=question,
        model_answer=model_answer,
        steps_block=steps_block,
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                'temperature': 0.1,
                'num_ctx': 8192,
                'num_predict': 2048,
                'stop': _finalize_role.stop,
            }
        )
    except Exception as e:
        logger.error(f"Сбой при подключении к LLM на шаге вынесения вердикта: {e}")
        return None, None

    raw_content = response['message'].get('content', '')
    done_reason = _get_done_reason(response)
    logger.info(f"finalize_raw_content: {raw_content}")
    logger.info(f"finalize_done_reason: {done_reason}")
    return raw_content, done_reason


def _clean_and_parse_finalize(raw_text: str, done_reason: str, logger,
                               steps_present: bool, steps_truncated: bool):
    """Разбирает ответ: <verdict>/<feedback>/<analysis>."""
    text = raw_text.strip()
    if not text.startswith("<verdict>"):
        text = _finalize_role.prefill + text

    was_truncated = done_reason == 'length'
    if was_truncated:
        logger.warning("Генерация оборвалась по num_predict на шаге вынесения вердикта (шаг 2).")

    defaults = {
        "verdict": "<verdict>FALSE</verdict>",
        "feedback": "<feedback>Ответ модели-верификатора не содержал этот блок.</feedback>",
        "analysis": "<analysis>Ответ модели-верификатора не содержал этот блок.</analysis>",
    }
    for tag in ("verdict", "feedback", "analysis"):
        open_t, close_t = f"<{tag}>", f"</{tag}>"
        if open_t in text and close_t not in text:
            # Штатный случай (done_reason == 'stop') -> закрывающий тег
            # вырезан стоп-последовательностью, просто дописываем его.
            text = text.rstrip() + f"\n{close_t}"
        elif open_t not in text:
            if done_reason not in ('length',):
                was_truncated = True
            text = text.rstrip() + "\n" + defaults[tag]

    verdict_raw = _extract_tag("verdict", text) or ""
    analysis = _extract_tag("analysis", text) or "Модель не заполнила описание."
    feedback = _extract_tag("feedback", text) or "Решение требует проверки."

    is_correct = "true" in verdict_raw.strip().lower()

    if not steps_present and is_correct:
        is_correct = False
        analysis = (
            analysis + " [Внимание: пошаговая проверка не дала ни одного шага, "
            "вердикт TRUE не подтверждён.]"
        ).strip()

    return {
        "is_correct_solution": is_correct,
        "analysis_thoughts": analysis,
        "feedback_for_solver": feedback,
        "is_truncated": was_truncated or steps_truncated,
    }


def call_llm_verifier(question: str, model_answer: str, logger):
    """
    Пошаговая проверка решения студента в два вызова LLM:
      1) разбор решения студента на шаги (OK/ERROR на каждый);
      2) короткая агрегация этого разбора в итоговый вердикт.

    разделение сделано, чтобы обрыв генерации по num_predict на
    шаге 1 не приводил к потере вердикта — шаг 2 почти
    всегда укладывается в маленький бюджет, так как ничего заново не
    анализирует, а только суммирует уже готовый результат.

    обрыв по num_predict определяется через done_reason из ответа Ollama
    """
    steps, steps_truncated = _run_stepwise_check(question, model_answer, logger)
    steps_block = _render_steps_block(steps, steps_truncated)

    raw_content, done_reason = _run_finalize(question, model_answer, steps_block, logger)

    if raw_content is None:
        return {
            "is_correct_solution": False,
            "analysis_thoughts": "Сбой работы экспертной LLM на шаге вынесения вердикта.",
            "feedback_for_solver": "Ошибка верификатора.",
            "steps": steps,
            "is_truncated": True,
        }

    result = _clean_and_parse_finalize(
        raw_content, done_reason, logger,
        steps_present=bool(steps),
        steps_truncated=steps_truncated,
    )
    result["steps"] = steps

    logger.info(
        f"VERDICT (LLM, без ground_truth): {result['is_correct_solution']} "
        f"| is_truncated={result['is_truncated']}"
    )
    return result