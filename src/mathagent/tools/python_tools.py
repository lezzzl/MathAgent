import json
import re
import time
from typing import Any

from langchain_core.tools import BaseTool, tool

from mathagent.tools.python_executor import PythonExecutor


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

REACT_V2_PYTHON_DESCRIPTION = """Execute a complete self-contained Python script for mathematical computation or verification.

Use this same tool both for a new computation and for correcting a previous
failed Python execution. Every invocation starts in a fresh process. Always
provide the complete corrected script, never a patch, and never assume that
imports, variables, or definitions from an earlier invocation still exist.

Before writing code, formulate the mathematical model. Use SymPy to implement
or verify a justified model, especially for exact arithmetic, symbolic
manipulation, enumeration, and independent checks. Do not blindly send a large
or poorly formulated geometry or combinatorics system to `sp.solve`.

Inside the script:
1. Import SymPy with `import sympy as sp`.
2. Prefer exact symbolic computation over floating-point approximations.
3. Write the mathematical reasoning as comments inside the code.
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
    return {
        "stdout": compact_stdout(execution["stdout"]),
        "stderr": "\n".join(stderr_lines[-STDERR_TAIL_LINES:]),
        "error_type": execution["error_type"],
        "returncode": execution["returncode"],
        "timeout": execution["timeout"],
        "latency_seconds": execution["latency_seconds"],
    }


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
