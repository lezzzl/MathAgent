"""Инструменты агента и их описания для систем-промпта.

Идея: модель явно знает свой инструментарий (перечислен в систем-промпте как
JSON: что делает инструмент + аргумент), и может ПРОВЕРЯТЬ свои утверждения на
конкретных примерах — как тесты для кода. Базовый инструмент — run_python:
модель пишет скрипт, мы исполняем его в отдельном процессе с таймаутом и
возвращаем вывод.

Формат вызова (prompt-based ReAct) — огороженный блок с именем инструмента:

    ```run_python
    for n in range(1, 20):
        assert n*n >= n
    print("ok")
    ```

Тело блока целиком уходит в инструмент как единственный аргумент.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

# Предел вывода инструмента, чтобы не раздувать контекст
MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class Tool:
    """Описание инструмента: имя, что делает, что кладётся в тело блока, исполнитель."""

    name: str
    description: str
    argument: str  # что модель кладёт в тело ```<name> ... ``` блока
    run: Callable[[str], str]


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Исполнить Python-код в изолированном процессе и вернуть stdout+stderr.

    ВНИМАНИЕ: это НЕ песочница — код выполняется в подпроцессе с таймаутом
    (флаг -I отключает пользовательские sitecustomize/env). Достаточно для
    доверенной локальной среды и математических проверок; не запускать на
    непроверенном вводе.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"(превышен таймаут {timeout:.0f}s — упрости проверку или сократи перебор)"
    output = (proc.stdout or "") + (proc.stderr or "")
    output = output.strip()
    if not output:
        return "(пустой вывод — не забудь print(...) для результата)"
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n…(вывод обрезан)"
    return output


# Скрипт символьной проверки тождества (исполняется через run_python ради таймаута).
# Переменная `raw` со строкой выражения впрыскивается заголовком в sympy_check().
_SYMPY_CHECK_SCRIPT = r"""
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)
T = standard_transformations + (implicit_multiplication_application,)
raw = raw.replace("^", "**").replace("==", "=")
if raw.count("=") != 1:
    print("ошибка: дай ровно одно равенство вида  LHS = RHS")
    raise SystemExit
left, right = raw.split("=")
try:
    L = parse_expr(left, transformations=T)
    R = parse_expr(right, transformations=T)
except Exception as exc:
    print("ошибка разбора выражения:", exc)
    raise SystemExit
diff = sp.simplify(L - R)
if diff == 0:
    print("ВЕРНО: тождество (LHS - RHS упрощается до 0)")
else:
    import random
    syms = sorted(diff.free_symbols, key=str)
    hit = None
    for _ in range(60):
        subs = {s: random.randint(1, 6) for s in syms}
        try:
            val = complex(diff.subs(subs))
        except Exception:
            continue
        if abs(val) > 1e-9:
            hit = (subs, val)
            break
    if hit:
        pretty = ", ".join(f"{s}={v}" for s, v in hit[0].items()) or "(без переменных)"
        print(f"НЕВЕРНО: не тождество. При {pretty}  LHS-RHS = {hit[1].real:.6g}")
    else:
        print(f"НЕ ТОЖДЕСТВО: упрощённая разность = {diff} (контрпример не подобран)")
"""


def sympy_check(expr: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Символьно проверить, что равенство LHS = RHS — тождество (верно при всех значениях).

    Сильнее численного перебора: `sin(x)^2+cos(x)^2 = 1` подтвердится точно.
    Логика (simplify(L-R)==0, иначе подбор контрпримера) спрятана внутрь.
    """
    header = f"raw = {expr.strip()!r}\n"
    return run_python(header + _SYMPY_CHECK_SCRIPT, timeout=timeout)


# Скрипт символьного решения уравнения/системы (переменная `raw` — заголовком)
_SOLVE_SCRIPT = r"""
import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)
T = standard_transformations + (implicit_multiplication_application,)
text = raw.replace("^", "**").replace("==", "=")
equations = []
for part in re.split(r"[;\n]", text):
    part = part.strip()
    if not part:
        continue
    try:
        if "=" in part:
            left, right = part.split("=", 1)
            equations.append(sp.Eq(parse_expr(left, transformations=T),
                                   parse_expr(right, transformations=T)))
        else:  # выражение без '=' считаем равным нулю
            equations.append(sp.Eq(parse_expr(part, transformations=T), 0))
    except Exception as exc:
        print("ошибка разбора:", part, "->", exc)
        raise SystemExit
symbols = sorted(set().union(*[e.free_symbols for e in equations]), key=str)
if not symbols:
    print("нет переменных для решения")
    raise SystemExit
try:
    solutions = sp.solve(equations, symbols, dict=True)
except Exception as exc:
    print("sympy.solve не справился:", exc)
    raise SystemExit
if not solutions:
    print(f"решений не найдено (переменные: {', '.join(map(str, symbols))})")
else:
    print(f"переменные: {', '.join(map(str, symbols))}; решений: {len(solutions)}")
    for i, sol in enumerate(solutions, 1):
        pretty = ", ".join(f"{k}={v}" for k, v in sol.items())
        print(f"  {i}) {pretty}")
"""


def solve_equation(equation: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Символьно решить уравнение или систему относительно её переменных.

    Одно уравнение — `x^2 - 5x + 6 = 0`; систему разделяй `;` или переносом строки.
    Выражение без `=` считается равным нулю.
    """
    header = f"raw = {equation.strip()!r}\n"
    return run_python(header + _SOLVE_SCRIPT, timeout=timeout)


# Скрипт распознавания замкнутой формы по числу (переменная `raw` — заголовком)
_CLOSED_FORM_SCRIPT = r"""
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)
T = standard_transformations + (implicit_multiplication_application,)
try:
    value = parse_expr(raw.replace("^", "**"), transformations=T).evalf(40)
    number = sp.Float(value, 40)
except Exception as exc:
    print("не смог вычислить число из ввода:", exc)
    raise SystemExit
constants = [sp.pi, sp.E, sp.EulerGamma, sp.sqrt(2), sp.sqrt(3), sp.sqrt(5),
             sp.log(2), sp.log(3), sp.GoldenRatio]
candidates = []
for extra in ([], constants):
    try:
        guess = sp.nsimplify(number, extra or None, rational=False)
    except Exception:
        continue
    if guess is not None and abs(sp.Float(guess.evalf(40)) - number) < 1e-12:
        candidates.append(sp.simplify(guess))
seen, uniq = set(), []
for c in candidates:
    key = str(c)
    if key not in seen and not c.is_number is False and key != str(number):
        seen.add(key)
        uniq.append(c)
print(f"число ≈ {sp.N(number, 15)}")
if uniq:
    for c in uniq:
        print(f"  замкнутая форма: {c}   (числено {sp.N(c, 15)})")
else:
    print("  замкнутой формы среди типичных констант не нашлось "
          "(возможно, это просто рациональное/десятичное значение)")
"""


def closed_form(value: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Распознать замкнутую форму числа: десятичное → π²/6, √2, ln2 и т.п.

    Полезно, чтобы получить точный ответ из численного значения или сверить его.
    """
    header = f"raw = {value.strip()!r}\n"
    return run_python(header + _CLOSED_FORM_SCRIPT, timeout=timeout)


# Реестр доступных инструментов. Добавляя новый — просто впиши сюда Tool(...).
TOOLS: dict[str, Tool] = {
    "run_python": Tool(
        name="run_python",
        description=(
            "Выполнить Python-код и вернуть его вывод. Используй, чтобы ПРОВЕРИТЬ "
            "утверждение на конкретных примерах: перебрать случаи, посчитать "
            "значение, найти контрпример. Стандартная библиотека доступна "
            "(math, itertools, fractions, sympy)."
        ),
        argument="Python-скрипт; печатай результат через print(). Один вызов = один скрипт.",
        run=run_python,
    ),
    "sympy_check": Tool(
        name="sympy_check",
        description=(
            "Символьно проверить, что равенство — ТОЖДЕСТВО (верно при всех значениях "
            "переменных): алгебраические преобразования, тригонометрия, упрощения. "
            "Точнее численного перебора. Если это не тождество — вернёт контрпример."
        ),
        argument="Одно равенство LHS = RHS, напр. `sin(x)^2 + cos(x)^2 = 1` или `(a+b)^2 = a^2+2*a*b+b^2`.",
        run=sympy_check,
    ),
    "solve_equation": Tool(
        name="solve_equation",
        description=(
            "Символьно решить уравнение или систему относительно её переменных "
            "(sympy.solve). Возвращает все решения. Надёжнее ручного решения."
        ),
        argument="Уравнение `x^2 - 5x + 6 = 0`; систему раздели `;`. Без `=` считается `=0`.",
        run=solve_equation,
    ),
    "closed_form": Tool(
        name="closed_form",
        description=(
            "Распознать замкнутую форму по числовому значению (десятичное → π²/6, √2, "
            "ln2, φ и т.п.). Используй, чтобы получить ТОЧНЫЙ ответ из численного "
            "приближения или сверить финальный ответ."
        ),
        argument="Число или выражение, дающее число, напр. `1.6449340668` или `2*atan(1)`.",
        run=closed_form,
    ),
}


def render_tools_block(tools: dict[str, Tool]) -> str:
    """Отрисовать список инструментов как JSON для вставки в систем-промпт."""
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "argument": tool.argument,
        }
        for tool in tools.values()
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
