import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PythonExecutionResult:
    """Результат запуска Python-кода."""

    stdout: str
    stderr: str
    returncode: int | None
    timeout: bool

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Возвращает результат в формате для state графа."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timeout": self.timeout,
        }


class PythonExecutor:
    """Запускает Python-код в отдельном процессе."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def run(self, code: str) -> PythonExecutionResult:
        """Запускает код и возвращает stdout, stderr, returncode и timeout."""
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        if code.endswith("```"):
            code = code[: -len("```")].strip()
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as file:
            file.write(code)
            code_path = file.name

        try:
            result = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return PythonExecutionResult(
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                timeout=False,
            )
        except subprocess.TimeoutExpired as exc:
            return PythonExecutionResult(
                stdout=_clean_output(exc.stdout),
                stderr=_clean_output(exc.stderr),
                returncode=None,
                timeout=True,
            )
        finally:
            Path(code_path).unlink(missing_ok=True)


def _clean_output(output: str | bytes | None) -> str:
    """Приводит stdout/stderr из subprocess к строке."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace").strip()
    return output.strip()
