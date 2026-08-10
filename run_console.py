"""Компактный вывод прогона: одна строка на задачу вместо простыни.

Зачем
-----
Узлы графа печатают подробности напрямую через `print` — промпты, размышления,
счётчики токенов, вердикты. На 30 задачах это читаемо, на 400 (IMO) — нет:
интересная строка теряется среди десятков тысяч.

Модуль решает это, НЕ трогая ни одного `print` в пайплайне:

* подробный вывод перехватывается и складывается в буфер **на поток**, потому что
  раннер решает задачи в ThreadPoolExecutor и общий `redirect_stdout` перемешал бы
  вывод разных задач (и был бы не потокобезопасен);
* в терминал уходит одна строка на задачу;
* всё, что похоже на предупреждение или ошибку, из буфера всё равно печатается —
  отдельными строками после итоговой, чтобы не потерялось.

Данные при этом не теряются: полный текст генераций и вердиктов пишет
`trajectory.py`, а метрики и ответы — JSONL прогона. Тихий режим гасит только
дубль этого в терминал.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Цвет
# ---------------------------------------------------------------------------
# colorama нужна только старым консолям Windows; на Linux хватает голых ANSI.
# Её отсутствие не должно ронять прогон, поэтому импорт мягкий.
try:  # pragma: no cover - зависит от окружения
    import colorama

    colorama.just_fix_windows_console()
except Exception:  # noqa: BLE001
    colorama = None

_ANSI = {
    "green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
    "cyan": "\033[36m", "grey": "\033[90m", "bold": "\033[1m", "reset": "\033[0m",
}

# По умолчанию цвет включается только для терминала. Прогоны идут через
# `| tee run.log`, и в файле escape-последовательности только мешают читать.
_COLOR_ENABLED = sys.stdout.isatty()


def configure_color(mode: str) -> None:
    """mode: auto | always | never."""
    global _COLOR_ENABLED
    _COLOR_ENABLED = {"always": True, "never": False}.get(mode, sys.stdout.isatty())


def paint(text: str, *styles: str) -> str:
    if not _COLOR_ENABLED or not styles:
        return text
    return "".join(_ANSI.get(s, "") for s in styles) + text + _ANSI["reset"]


# ---------------------------------------------------------------------------
# Перехват подробного вывода
# ---------------------------------------------------------------------------
_local = threading.local()
_print_lock = threading.Lock()
QUIET = False

# Строки, которые нельзя терять даже в тихом режиме: явные пометки проблем и
# диагностические маркеры пайплайна.
_ALERT_RE = re.compile(
    r"⚠️|❌|\[ABORT\]|\[TOOL PARSING BROKEN\]|СЕТЕВОЙ СБОЙ|ОШИБКА|"
    r"\bERROR\b|Traceback|не распозна|обрыв|"
    # Ниже — маркеры 9B-пайплайна. Они печатаются без ⚠️, поэтому без явного
    # перечисления тихий режим их проглатывал. Оба означают, что роль упёрлась
    # в num_predict: размышления съели лимит и content пришёл пустым — это
    # главный режим отказа reasoning-модели, терять его в логе нельзя.
    r"\[THINKING OVERRUN\]|\[TRUNCATED\]",
    re.IGNORECASE,
)


def _buffer() -> Optional[List[str]]:
    return getattr(_local, "buf", None)


def _gated_print(*args: Any, **kwargs: Any) -> None:
    """Замена `print` внутри модулей пайплайна.

    В обычном режиме — обычная печать. В тихом — строка уходит в буфер задачи,
    откуда потом достаются только тревожные строки.
    """
    buf = _buffer()
    if not QUIET or buf is None:
        with _print_lock:
            print(*args, **kwargs)
        return
    sep = kwargs.get("sep", " ")
    buf.append(sep.join(str(a) for a in args))


def install(modules: List[Any], quiet: bool) -> None:
    """Подменяет `print` в перечисленных модулях.

    Работает потому, что имя `print` ищется сначала в globals модуля и только
    потом в builtins. Так не нужно править полсотни вызовов в графе, и правка
    полностью обратима.
    """
    global QUIET
    QUIET = quiet
    if not quiet:
        return
    for mod in modules:
        if mod is not None:
            mod.print = _gated_print  # type: ignore[attr-defined]


def begin_task() -> None:
    _local.buf = []


def take_alerts(limit: int = 6) -> List[str]:
    """Забирает из буфера строки, которые стоит показать, и очищает буфер."""
    buf = _buffer() or []
    _local.buf = []
    alerts = [l.strip() for l in buf if l.strip() and _ALERT_RE.search(l)]
    # Дубликаты в пределах задачи не нужны: одна и та же жалоба повторяется
    # на каждом витке восстановления.
    seen, out = set(), []
    for line in alerts:
        if line not in seen:
            seen.add(line)
            out.append(line)
        if len(out) >= limit:
            out.append(f"… ещё {len(alerts) - limit} похожих строк, полностью — в траектории")
            break
    return out


# ---------------------------------------------------------------------------
# Проверка ответа для строки состояния
# ---------------------------------------------------------------------------
_verify_fn: Optional[Callable[[str, str], Optional[bool]]] = None


def _load_verifier() -> Optional[Callable[[str, str], Optional[bool]]]:
    """Ленивая и необязательная сверка ответа для PASS/FAIL в терминале.

    Это ТОЛЬКО индикатор. Итоговая точность по-прежнему считается отдельным
    процессом после прогона (`verify_answers.py`), и если здесь что-то пойдёт не
    так, статус станет «?», но прогон не пострадает.
    """
    global _verify_fn
    if _verify_fn is not None:
        return _verify_fn
    try:
        import logging

        from verify_answers import extract_and_verify

        # math_verify на каждый разбор пишет WARNING про отключённый таймаут.
        # На 400 задачах это 800 строк ровно в том выводе, который мы чистим.
        # Таймаут отключён намеренно (на Windows его multiprocessing-обёртка не
        # работает), так что предупреждение здесь чистый шум.
        for name in ("math_verify", "math_verify.parser", "math_verify.grader",
                     "verify_answers"):
            logging.getLogger(name).setLevel(logging.ERROR)

        def _check(answer: str, gt: str) -> Optional[bool]:
            try:
                return bool(extract_and_verify(answer, gt, None)["is_correct"])
            except Exception:  # noqa: BLE001
                return None

        _verify_fn = _check
    except Exception:  # noqa: BLE001
        _verify_fn = lambda answer, gt: None  # noqa: E731
    return _verify_fn


def status_of(answer: Optional[str], ground_truth: str, error: Optional[str]) -> str:
    if error:
        return "ERROR"
    if not str(answer or "").strip():
        return "EMPTY"
    verdict = _load_verifier()(str(answer), str(ground_truth))
    if verdict is None:
        return "?"
    return "PASS" if verdict else "FAIL"


_STATUS_STYLE = {
    "PASS": ("green", "bold"), "FAIL": ("red",), "ERROR": ("red", "bold"),
    "EMPTY": ("yellow",), "?": ("grey",),
}


def _short(value: Any, width: int = 22) -> str:
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if not text:
        return "—"
    return text if len(text) <= width else text[: width - 1] + "…"


def task_line(index: int, total: int, status: str, answer: Any, ground_truth: Any,
              tokens: int, seconds: float, extra: str = "") -> str:
    width = len(str(total))
    head = paint(f"[{index:0{width}d}/{total}]", "cyan")
    tag = paint(f"{status:<5}", *_STATUS_STYLE.get(status, ()))
    tail = f" | {extra}" if extra else ""
    return (f"{head} {tag} | Model: {_short(answer)} | GT: {_short(ground_truth)} "
            f"| Tokens: {tokens:,} | Time: {seconds:.1f}s{tail}")


def emit(line: str, alerts: List[str], error: Optional[str] = None) -> None:
    """Печатает строку задачи и, отдельными строками, всё тревожное."""
    with _print_lock:
        print(line, flush=True)
        if error:
            print(paint(f"    ERROR: {error}", "red", "bold"), flush=True)
        for alert in alerts:
            print(paint(f"    ! {_short(alert, 160)}", "yellow"), flush=True)
        if error or alerts:
            sys.stdout.flush()
