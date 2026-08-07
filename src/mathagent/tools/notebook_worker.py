import contextlib
import io
import json
import sys
import traceback
from typing import Any


MAX_VISIBLE_NAMES = 100


def visible_names(namespace: dict[str, Any]) -> tuple[list[str], bool]:
    """Возвращает ограниченный список пользовательских имён notebook-сессии."""
    names = sorted(name for name in namespace if not name.startswith("_"))
    return names[:MAX_VISIBLE_NAMES], len(names) > MAX_VISIBLE_NAMES


def execute_cell(
    code: str,
    namespace: dict[str, Any],
    cell_index: int,
) -> dict[str, Any]:
    """Выполняет одну ячейку, сохраняя namespace для следующих вызовов."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    returncode = 0
    error_type: str | None = None
    message: str | None = None
    line: int | None = None
    offset: int | None = None
    source_line: str | None = None
    names_before = set(namespace)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            compiled = compile(code, f"<notebook-cell-{cell_index}>", "exec")
            exec(compiled, namespace, namespace)
        except BaseException as exc:
            returncode = 1
            error_type = type(exc).__name__
            message = str(exc)
            extracted = traceback.extract_tb(exc.__traceback__)
            if extracted:
                frame = extracted[-1]
                line = frame.lineno
                source_line = frame.line
                if not source_line and 1 <= line <= len(code.splitlines()):
                    source_line = code.splitlines()[line - 1]
            traceback.print_exc()
    available_names, names_truncated = visible_names(namespace)
    defined_names = sorted(
        name for name in set(namespace) - names_before if not name.startswith("_")
    )[:MAX_VISIBLE_NAMES]
    return {
        "success": returncode == 0,
        "stdout": stdout.getvalue().strip(),
        "stderr": stderr.getvalue().strip(),
        "error_type": error_type,
        "message": message,
        "line": line,
        "offset": offset,
        "source_line": source_line,
        "returncode": returncode,
        "timeout": False,
        "available_names": available_names,
        "defined_names": defined_names,
        "names_truncated": names_truncated,
    }


def main() -> None:
    """Читает JSON-запросы из stdin и возвращает результаты ячеек в stdout."""
    namespace: dict[str, Any] = {"__name__": "__main__"}
    cell_index = 0
    for request_line in sys.stdin:
        request = json.loads(request_line)
        code = request["code"]
        result = execute_cell(code, namespace, cell_index)
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        cell_index += 1


if __name__ == "__main__":
    main()
