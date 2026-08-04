import json
import re
import time
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from mathagent.tools.python_executor import NotebookSessionManager, PythonExecutor


STDOUT_LIMIT_CHARS = 6000
STDOUT_HEAD_CHARS = 1000
STDERR_TAIL_LINES = 30
EXCEPTION_NAME = re.compile(
    r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Interrupt))(?=:|$)"
)

LEGACY_PYTHON_DESCRIPTION = """Execute a complete self-contained Python script for a mathematical task.

The optional purpose briefly states what the execution should establish.
Inside the Python script:
1. Import SymPy with `import sympy as sp`.
2. Use exact symbolic computation whenever possible.
3. Write the mathematical reasoning as comments inside the code.
4. Clearly define variables, constraints, and the goal.
5. Use meaningful variable names.
6. Add assertions or checks for important constraints when possible.
"""

REACT_V2_PYTHON_DESCRIPTION = """Execute a Python notebook cell for mathematical computation or verification.

Use this same tool both for a new computation and for correcting a previous
failed Python cell. Calls for one task share a persistent notebook namespace, so
imports, variables, functions, and symbolic objects remain available to later
cells. Build on valid existing state instead of repeating expensive work.

A normal Python exception does not reset the session. Fix it with another cell;
variables created before the error may still exist. A timeout or worker crash
returns `session_reset=true` and destroys the namespace. After a reset, rebuild
all state needed by later cells.

The context is a compact record of the mathematical reasoning already
established, including the current approach, relevant assumptions, reductions,
and constraints. Preserve the information needed by the next agent iteration
without copying hidden chain-of-thought or the full conversation.

Before writing code, formulate the mathematical model. Use SymPy to implement
or verify a justified model, especially for exact arithmetic, symbolic
manipulation, enumeration, and independent checks. Do not blindly send a large
or poorly formulated geometry or combinatorics system to `sp.solve`.

Inside the notebook session:
1. Import SymPy with `import sympy as sp` when first needed; later cells may reuse it.
2. Prefer exact symbolic computation over floating-point approximations.
3. Use Program-of-Thought style: write the mathematical reasoning, deductions,
   modeling choices, and expected meaning of the output as comments alongside
   the executable code.
4. Clearly define variables, constraints, and the goal.
5. Use meaningful variable names.
6. Add assertions or independent checks for important constraints when possible.
7. Print only concise observations and results needed by the agent.
"""


def compact_stdout(stdout: str) -> str:
    """Оставляет начало и преимущественно конец длинного stdout."""
    if len(stdout) <= STDOUT_LIMIT_CHARS:
        return stdout

    marker = "\n... [stdout truncated] ...\n"
    tail_chars = STDOUT_LIMIT_CHARS - STDOUT_HEAD_CHARS - len(marker)
    return f"{stdout[:STDOUT_HEAD_CHARS]}{marker}{stdout[-tail_chars:]}"


def get_error_type(execution: dict[str, Any]) -> str | None:
    """Определяет тип execution-ошибки для компактного observation."""
    if execution["timeout"]:
        return "TimeoutError"
    if execution["returncode"] == 0:
        return None

    for line in reversed(execution["stderr"].splitlines()):
        match = EXCEPTION_NAME.search(line.strip())
        if match:
            return match.group(1)
    return "ExecutionError"


def compact_execution(execution: dict[str, Any]) -> dict[str, Any]:
    """Формирует короткий Python observation для контекста модели."""
    stderr_lines = execution["stderr"].splitlines()
    compact = {
        "stdout": compact_stdout(execution["stdout"]),
        "stderr": "\n".join(stderr_lines[-STDERR_TAIL_LINES:]),
        "error_type": execution["error_type"],
        "returncode": execution["returncode"],
        "timeout": execution["timeout"],
        "latency_seconds": execution["latency_seconds"],
    }
    if "session_reset" in execution:
        compact["session_reset"] = execution["session_reset"]
    return compact


def execute_code(executor: PythonExecutor, code: str) -> tuple[str, dict[str, Any]]:
    """Исполняет код и разделяет компактный content и полный artifact."""
    started = time.perf_counter()
    execution = executor.run(code).to_dict()
    execution["latency_seconds"] = round(time.perf_counter() - started, 3)
    execution["error_type"] = get_error_type(execution)
    return json.dumps(compact_execution(execution), ensure_ascii=False), execution


def create_python_tool(
    timeout: float,
    description: str = LEGACY_PYTHON_DESCRIPTION,
) -> BaseTool:
    """Создаёт tool для запуска полного Python/SymPy-скрипта."""
    executor = PythonExecutor(timeout=timeout)

    @tool(
        "python",
        description=description,
        response_format="content_and_artifact",
    )
    def python_tool(
        code: str,
        purpose: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Выполняет переданный полный Python-скрипт в новом процессе."""
        del purpose
        return execute_code(executor, code)

    return python_tool


def create_notebook_python_tool(manager: NotebookSessionManager) -> BaseTool:
    """Создаёт notebook-like Python-tool с отдельной сессией на задачу."""

    @tool(
        "python",
        description=REACT_V2_PYTHON_DESCRIPTION,
        response_format="content_and_artifact",
    )
    def python_tool(
        code: str,
        context: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> tuple[str, dict[str, Any]]:
        """Выполняет ячейку, сохраняя namespace, цель и context задачи."""
        del context
        session_id = state.get("python_session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Notebook Python tool requires a session id")
        started = time.perf_counter()
        execution = manager.run(session_id, code)
        execution["latency_seconds"] = round(time.perf_counter() - started, 3)
        execution["error_type"] = get_error_type(execution)
        return (
            json.dumps(compact_execution(execution), ensure_ascii=False),
            execution,
        )

    return python_tool


def create_repair_tool(timeout: float) -> BaseTool:
    """Создаёт tool для замены последнего неуспешного Python-скрипта."""
    executor = PythonExecutor(timeout=timeout)

    @tool("repair", response_format="content_and_artifact")
    def repair_tool(corrected_code: str) -> tuple[str, dict[str, Any]]:
        """Repair the most recent failed Python execution.

        Provide a complete self-contained corrected script, not a patch or an
        explanation. Preserve valid mathematical modeling from the failed script
        when possible and fix its syntax, name, runtime, or timeout problem.
        """
        return execute_code(executor, corrected_code)

    return repair_tool
