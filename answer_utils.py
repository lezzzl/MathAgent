"""Извлечение и нормализация финального ответа из текста решения.

Вынесено в отдельный модуль, потому что этим занимаются и пошаговый солвер,
и режим self-consistency, и голосование по кандидатам — раньше в каждом месте
была своя регулярка, и все они по-разному ломались на вложенных скобках.
"""

import re
from collections import Counter
from typing import Iterable, List, Optional, Tuple

_BOXED_MARKERS = ("\\boxed", "\\fbox")

# Фоллбек, когда \boxed{} в тексте нет вообще: "the answer is 42".
_FINAL_ANSWER_RE = re.compile(
    r"(?i)(?:final answer|the answer is|answer is|answer:|result is)"
    r"[^0-9\-]{0,60}(-?[0-9]+(?:\.[0-9]+)?)"
)

_TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|mbox|textbf)\s*\{([^{}]*)\}")
_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right|!|,|;|:|\s)")


def _extract_braced(text: str, open_idx: int) -> Optional[str]:
    """Возвращает содержимое {...}, начиная с открывающей скобки по open_idx.

    Считает вложенность, поэтому \\boxed{\\frac{1}{2}} извлекается целиком,
    а не обрезается на первой закрывающей скобке, как это делала старая
    регулярка [^}]* в langgraph_math_solver.extract_answer.
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    for i in range(open_idx, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    # Скобка не закрыта — модель оборвалась на лимите токенов.
    # Берём всё до конца, лучше частичный ответ, чем никакого.
    return text[open_idx + 1 :]


def iter_boxed(text: str) -> Iterable[str]:
    """Перебирает все \\boxed{...} / \\fbox{...} в порядке появления."""
    if not text:
        return
    for marker in _BOXED_MARKERS:
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx == -1:
                break
            after = idx + len(marker)
            while after < len(text) and text[after] in " \t":
                after += 1
            if after < len(text) and text[after] == "{":
                content = _extract_braced(text, after)
                if content is not None:
                    yield content
                start = after + 1
            else:
                # \boxed 42 без скобок
                m = re.match(r"\s*(-?[0-9]+)", text[after:])
                if m:
                    yield m.group(1)
                start = after + 1


def extract_boxed(text: str) -> Optional[str]:
    """Последний \\boxed{...} в тексте, либо None."""
    found = [c.strip() for c in iter_boxed(text or "") if c.strip()]
    return found[-1] if found else None


def extract_answer(text: str) -> Optional[str]:
    """Финальный ответ: сначала \\boxed{...}, потом фраза 'the answer is ...'."""
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Приводит ответ к канонической форме для сравнения и голосования.

    Задача — чтобы '079', '79', '$79$', '79.0' и '\\text{79}' попали в одну
    корзину при majority voting. Возвращает None для пустого ввода.
    """
    if answer is None:
        return None
    text = answer.strip()
    if not text:
        return None

    # Разворачиваем \text{...} и снимаем латеховый мусор.
    for _ in range(3):
        new_text = _TEXT_WRAPPER_RE.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text
    text = _LEFT_RIGHT_RE.sub(" ", text)
    text = text.replace("$", "").replace("\\%", "").replace("%", "")
    text = text.replace("{,}", "").replace(",", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\s+", "", text).strip().rstrip(".")
    if not text:
        return None

    # Целое число в любом виде записи: 079 -> 79, 79.0 -> 79, +79 -> 79.
    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
        return repr(value)
    except ValueError:
        pass

    # \frac{a}{b} -> a/b, чтобы совпадали разные записи одной дроби.
    frac = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    if frac:
        return f"{frac.group(1)}/{frac.group(2)}"

    return text


def majority_vote(
    answers: Iterable[Optional[str]],
) -> Tuple[Optional[str], List[Tuple[str, int]]]:
    """Мажоритарное голосование по ответам сэмплов (self-consistency).

    Возвращает (победитель, распределение голосов по убыванию). Пустые и
    непарсящиеся ответы отбрасываются — они не должны «съедать» голос.
    При равенстве голосов побеждает тот, кто встретился раньше: Counter
    сохраняет порядок вставки, а most_common стабилен.
    """
    normalized = [n for n in (normalize_answer(a) for a in answers) if n]
    if not normalized:
        return None, []
    counter = Counter(normalized)
    ranked = counter.most_common()
    return ranked[0][0], ranked
