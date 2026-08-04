import contextlib
import io
import json
import sys
import traceback
from typing import Any


def execute_cell(
    code: str,
    namespace: dict[str, Any],
    cell_index: int,
) -> dict[str, str | int | bool | None]:
    """Выполняет одну ячейку, сохраняя namespace для следующих вызовов."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    returncode = 0
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            compiled = compile(code, f"<notebook-cell-{cell_index}>", "exec")
            exec(compiled, namespace, namespace)
        except BaseException:
            returncode = 1
            traceback.print_exc()
    return {
        "stdout": stdout.getvalue().strip(),
        "stderr": stderr.getvalue().strip(),
        "returncode": returncode,
        "timeout": False,
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
