from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest

from scripts.jobs import run_job, sandbox
from scripts.jobs.job_spec import JobSpec

LINUX_ONLY = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="уборка по маркеру в /proc работает только на Linux — на VM именно он",
)


def make_spec(command: str, argv: list[str], *, timeout_minutes: int = 1) -> JobSpec:
    return JobSpec(
        path=PurePosixPath("jobs/queue/ilya-demo.yml"),
        branch="Ilya/demo",
        command=command,
        gpus=1,
        timeout_minutes=timeout_minutes,
        commit_sha="0" * 40,
        argv=argv,
    )


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "mathagent"
    monkeypatch.setenv("MATHAGENT_HOME", str(directory))
    # Тесты запуска не должны зависеть от наличия bwrap на машине разработчика.
    monkeypatch.setenv("MATHAGENT_SANDBOX", "0")
    return directory


@pytest.fixture
def workspace(home: Path) -> Path:
    directory = home / "work" / "ilya-demo"
    (directory / "tmp").mkdir(parents=True)
    (directory / "results" / "runs" / "ilya-demo").mkdir(parents=True)
    return directory


# --- окружение ----------------------------------------------------------------


def test_environment_carries_no_secrets(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Окружение собирается с нуля: токен и ключ владельца машины в него не попадают."""
    monkeypatch.setenv("MATHAGENT_GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /home/teacher/.ssh/id_ed25519")

    spec = make_spec("python x.py", ["python", "x.py"])
    environment = run_job.build_environment(spec, workspace, "0")

    assert set(environment) == {
        "PATH", "HOME", "TMPDIR", "LANG",
        "CUDA_VISIBLE_DEVICES", "HF_HOME", "UV_CACHE_DIR", run_job.RUN_ID_ENV,
    }
    assert "secret-token" not in "".join(environment.values())
    assert "GIT_SSH_COMMAND" not in environment
    # HOME и TMPDIR внутри рабочего каталога — случайные записи остаются в нём.
    assert environment["HOME"] == str(workspace)
    assert environment["TMPDIR"] == str(workspace / "tmp")
    assert environment[run_job.RUN_ID_ENV] == "ilya-demo"


def test_sandbox_is_on_by_default(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изоляция — состояние по умолчанию, выключается только явным флагом."""
    monkeypatch.delenv("MATHAGENT_SANDBOX", raising=False)
    home.mkdir(parents=True, exist_ok=True)
    assert sandbox.is_enabled() is True

    (home / sandbox.DISABLE_FLAG_NAME).touch()
    assert sandbox.is_enabled() is False


def test_sandbox_hides_personal_credentials(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ключ и токен владельца машины подменяются пустым tmpfs."""
    secrets = tmp_path / "ssh"
    secrets.mkdir()
    monkeypatch.setenv("MATHAGENT_SENSITIVE", str(secrets))
    monkeypatch.setattr(sandbox, "probe", lambda: (True, "stub"))

    argv = sandbox.wrap(["python", "x.py"], workspace=workspace, writable=[])

    assert argv[0] == "bwrap"
    assert argv[1:4] == ["--ro-bind", "/", "/"]
    assert ["--tmpfs", str(secrets)] == argv[argv.index("--tmpfs"):argv.index("--tmpfs") + 2]
    # Записывать можно только в рабочий каталог запуска.
    assert ["--bind", str(workspace), str(workspace)] == argv[
        argv.index("--bind"):argv.index("--bind") + 3
    ]
    assert argv[-2:] == ["python", "x.py"]


def test_broken_sandbox_refuses_to_run(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Молча выполнить команду без изоляции хуже, чем не выполнить её вовсе."""
    monkeypatch.setenv("MATHAGENT_SANDBOX", "1")
    monkeypatch.setattr(sandbox, "probe", lambda: (False, "bwrap is not installed"))
    spec = make_spec("python x.py", ["python", "x.py"])

    with pytest.raises(sandbox.SandboxError, match="bwrap is not installed"):
        run_job.build_argv(spec, workspace)


# --- таймаут и уборка ---------------------------------------------------------


def test_timeout_kills_the_command(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(JobSpec, "timeout_seconds", property(lambda _self: 1.0))
    spec = make_spec(
        "python sleep.py", [sys.executable, "-c", "import time; time.sleep(120)"]
    )

    started = time.monotonic()
    returncode, status, _duration = run_job.execute(spec, workspace, "0")

    assert status == "timeout"
    assert returncode != 0
    # Ждём таймаут, а не полные две минуты команды.
    assert time.monotonic() - started < 30


@LINUX_ONLY
def test_cleanup_kills_children_that_escaped_the_process_group(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Забытый vLLM — главная причина занятой карты, поэтому setsid не спасает потомка."""
    monkeypatch.setattr(JobSpec, "timeout_seconds", property(lambda _self: 1.0))
    daemon = (
        "import os, sys, time\n"
        "if os.fork() > 0: sys.exit(0)\n"
        "os.setsid()\n"
        "while True: time.sleep(1)\n"
    )
    spec = make_spec("python daemon.py", [sys.executable, "-c", daemon])

    run_job.execute(spec, workspace, "0")

    assert run_job._marked_pids("ilya-demo") == []


@LINUX_ONLY
def test_cleanup_does_not_touch_another_run(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Уборка по маркеру точечная: параллельный запуск другого студента не задет."""
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        env={**os.environ, run_job.RUN_ID_ENV: "petya-other"},
    )
    try:
        run_job.terminate_run(None, "ilya-demo")
        assert other.poll() is None
    finally:
        other.kill()
        other.wait(timeout=10)


def test_result_file_records_the_outcome(workspace: Path, home: Path) -> None:
    """Диспетчер узнаёт итог запуска именно из этого файла."""
    from scripts.benchmarks.run_artifacts import read_json

    (home / "state").mkdir(parents=True, exist_ok=True)
    spec = make_spec("python x.py", ["python", "x.py"])

    run_job.write_result(spec, "failed", 3, 12.5)

    result = read_json(run_job.result_path("ilya-demo"))
    assert result["status"] == "failed"
    assert result["exit_code"] == 3
    assert result["duration_seconds"] == 12.5
