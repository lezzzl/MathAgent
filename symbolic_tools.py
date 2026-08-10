"""Символьные инструменты MathAgent как нативные LangChain-тулы.

Обёртки над mathagent.agent.tools (sympy_check / solve_equation / closed_form),
чтобы их можно было добавить в TOOLS командного агента (tool_generator_subgraph)
рядом с python_exec. Модель выбирает тул по DOCSTRING — поэтому описания здесь
подробные и на английском, как у python_exec.

Логика не дублируется: сами вычисления берём из mathagent.agent.tools.
"""

import sys
from pathlib import Path

from langchain_core.tools import tool

# mathagent живёт в src/ — добавляем его в путь (корневые модули агента этого не делают)
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mathagent.agent.tools import (  # noqa: E402
    closed_form as _closed_form,
    solve_equation as _solve_equation,
    sympy_check as _sympy_check,
)


@tool
def sympy_check(identity: str) -> str:
    """Symbolically check whether an equation is an IDENTITY (true for ALL values
    of its variables). Use this to verify an algebraic or trigonometric step you
    rely on — an expansion, factoring or simplification. It is more reliable than
    testing a few numbers by hand. Returns confirmation if it is an identity, or a
    concrete counterexample if it is not.

    Input: one equation "LHS = RHS", e.g.
        sin(x)^2 + cos(x)^2 = 1
        (a+b)^2 = a^2 + 2*a*b + b^2
    """
    return _sympy_check(identity)


@tool
def solve_equation(equation: str) -> str:
    """Solve an equation or system of equations EXACTLY for its variables
    (symbolic, via sympy). Use this instead of solving by hand to avoid algebra
    mistakes; it returns all solutions.

    Input: one equation such as "x^2 - 5*x + 6 = 0", or a system separated by ";".
    An expression without "=" is treated as "= 0". Example:
        x^2 - 5*x + 6 = 0
        a + b = 10; a - b = 4
    """
    return _solve_equation(equation)


@tool
def closed_form(value: str) -> str:
    """Recognize the exact CLOSED FORM of a numeric value (e.g. 1.6449... -> pi^2/6,
    1.41421... -> sqrt(2), 0.6931... -> ln(2), 1.61803... -> the golden ratio). Use
    this to turn a decimal result into an exact answer, or to double-check a final
    numeric answer.

    Input: a number, or an expression that evaluates to a number, e.g.
        1.6449340668
        2*atan(1)
    """
    return _closed_form(value)


# Список для удобного добавления в TOOLS командного агента
SYMBOLIC_TOOLS = [sympy_check, solve_equation, closed_form]
