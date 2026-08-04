import atexit
import json
import selectors
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


class NotebookPythonExecutor:
    """Хранит Python namespace задачи в отдельном persistent процессе."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def run(self, code: str) -> dict[str, str | int | bool | None]:
        """Выполняет ячейку в текущей сессии и сбрасывает её после timeout."""
        with self._lock:
            process = self._ensure_process()
            request = json.dumps({"code": _strip_code_fence(code)}, ensure_ascii=False)
            try:
                if process.stdin is None:
                    raise RuntimeError("Notebook worker stdin is unavailable")
                process.stdin.write(request + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, RuntimeError) as exc:
                self._stop_process()
                return _worker_failure(exc)

            if process.stdout is None:
                self._stop_process()
                return _worker_failure(RuntimeError("Notebook worker stdout is unavailable"))

            selector = selectors.DefaultSelector()
            try:
                selector.register(process.stdout, selectors.EVENT_READ)
                ready = selector.select(self.timeout)
            finally:
                selector.close()

            if not ready:
                self._stop_process()
                return {
                    "stdout": "",
                    "stderr": "Python notebook cell exceeded its timeout",
                    "returncode": None,
                    "timeout": True,
                    "session_reset": True,
                }

            response_line = process.stdout.readline()
            if not response_line:
                worker_stderr = self._read_worker_stderr(process)
                self._stop_process()
                return {
                    "stdout": "",
                    "stderr": worker_stderr or "Python notebook worker terminated",
                    "returncode": 1,
                    "timeout": False,
                    "session_reset": True,
                }

            try:
                result = json.loads(response_line)
            except json.JSONDecodeError as exc:
                self._stop_process()
                return _worker_failure(exc)
            result["session_reset"] = False
            return result

    def close(self) -> None:
        """Завершает persistent процесс и освобождает его ресурсы."""
        with self._lock:
            self._stop_process()

    def _ensure_process(self) -> subprocess.Popen[str]:
        """Возвращает живой worker или запускает новую чистую сессию."""
        if self._process is not None and self._process.poll() is None:
            return self._process
        worker_path = Path(__file__).with_name("notebook_worker.py")
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._process

    def _stop_process(self) -> None:
        """Останавливает worker, включая зависший процесс после timeout."""
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    @staticmethod
    def _read_worker_stderr(process: subprocess.Popen[str]) -> str:
        """Читает системную ошибку завершившегося worker-процесса."""
        if process.stderr is None or process.poll() is None:
            return ""
        return process.stderr.read().strip()


class NotebookSessionManager:
    """Изолирует persistent Python-сессии параллельных задач."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._sessions: dict[str, NotebookPythonExecutor] = {}
        self._lock = threading.Lock()
        atexit.register(self.close_all)

    def run(self, session_id: str, code: str) -> dict[str, Any]:
        """Выполняет ячейку в сессии указанной задачи."""
        with self._lock:
            executor = self._sessions.get(session_id)
            if executor is None:
                executor = NotebookPythonExecutor(timeout=self.timeout)
                self._sessions[session_id] = executor
        return executor.run(code)

    def close(self, session_id: str) -> None:
        """Закрывает одну задачу и удаляет её namespace."""
        with self._lock:
            executor = self._sessions.pop(session_id, None)
        if executor is not None:
            executor.close()

    def close_all(self) -> None:
        """Закрывает все оставшиеся сессии при завершении процесса runner."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for executor in sessions:
            executor.close()

    @property
    def active_session_count(self) -> int:
        """Возвращает число ещё не закрытых сессий для диагностики."""
        with self._lock:
            return len(self._sessions)


def _clean_output(output: str | bytes | None) -> str:
    """Приводит stdout/stderr из subprocess к строке."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace").strip()
    return output.strip()


def _strip_code_fence(code: str) -> str:
    """Убирает необязательный внешний Markdown-блок из Python-кода."""
    value = code.strip()
    if value.startswith("```python"):
        value = value[len("```python") :].strip()
    if value.endswith("```"):
        value = value[: -len("```")].strip()
    return value


def _worker_failure(exception: Exception) -> dict[str, str | int | bool | None]:
    """Возвращает структурированную ошибку аварийной потери worker-сессии."""
    return {
        "stdout": "",
        "stderr": f"{type(exception).__name__}: {exception}",
        "returncode": 1,
        "timeout": False,
        "session_reset": True,
    }
