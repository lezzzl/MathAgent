"""Выполнение одного запуска студента.

Процесс живёт дольше тика cron: диспетчер отсоединяет его через setsid и сразу
завершает тик. Аренда GPU наследуется дескрипторами и держится ровно столько,
сколько живёт этот процесс.

Команда студента запускается без shell, в собранном с нуля окружении и с
маркером MATHAGENT_RUN_ID. Маркер наследуется всеми потомками и позволяет убить
именно этот запуск: главная неприятность здесь — оставшийся после таймаута
vLLM-сервер, который продолжает держать карту.
"""

from __future__ import annotations

import argparse
import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks.run_artifacts import utc_now, write_json_atomic
from scripts.jobs import sandbox
from scripts.jobs.gpu_lease import adopt_descriptors
from scripts.jobs.job_spec import JobSpec, JobSpecError, git_output, load_spec_text
from scripts.jobs.paths import (
    ensure_layout,
    mathagent_home,
    repo_dir,
    state_dir,
    work_dir,
)

RUN_ID_ENV = "MATHAGENT_RUN_ID"

# Сигнал после таймаута: сначала даём завершиться штатно, потом убиваем.
GRACE_SECONDS = 10.0


def cache_dir(name: str) -> Path:
    """Общий кэш моделей и пакетов: без него каждый студент качает Qwen заново."""
    return mathagent_home() / "cache" / name


def result_path(run_id: str) -> Path:
    """Файл с итогом запуска. Его пишет только этот процесс."""
    return state_dir() / f"{run_id}.result.json"


def workspace_path(run_id: str) -> Path:
    return work_dir() / run_id


def results_dir(workspace: Path, run_id: str) -> Path:
    return workspace / "results" / "runs" / run_id


def load_spec_from_main(spec_path: PurePosixPath, repo: Path) -> JobSpec:
    """Перечитывает спеку из origin/main и валидирует её заново.

    Проверка на стороне PR не является доверенной границей: между ней и запуском
    проходит время, а сам запуск обязан опираться на то, что действительно лежит
    в main.
    """
    text = git_output(repo, "show", f"origin/main:{spec_path}")
    return load_spec_text(text, spec_path=spec_path, repo=repo)


def prepare_workspace(spec: JobSpec, repo: Path) -> Path:
    """Создаёт отдельную рабочую копию репозитория на нужном коммите."""
    workspace = workspace_path(spec.run_id)
    if workspace.exists():
        # Остатки прошлой попытки: worktree remove чистит и служебные записи git.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace)],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    workspace.parent.mkdir(parents=True, exist_ok=True)

    git_output(repo, "worktree", "add", "--detach", str(workspace), spec.commit_sha)
    (workspace / "tmp").mkdir(exist_ok=True)
    results_dir(workspace, spec.run_id).mkdir(parents=True, exist_ok=True)
    return workspace


def build_environment(spec: JobSpec, workspace: Path, devices: str) -> dict[str, str]:
    """Собирает окружение команды с нуля.

    Наследовать окружение диспетчера нельзя: в нём токен владельца машины и всё,
    что подтянул cron. HOME и TMPDIR указывают внутрь рабочего каталога, поэтому
    случайные записи (кэши, точки-файлы) остаются в пределах запуска.
    """
    huggingface_cache = cache_dir("huggingface")
    uv_cache = cache_dir("uv")
    for directory in (huggingface_cache, uv_cache):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workspace),
        "TMPDIR": str(workspace / "tmp"),
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": devices,
        "HF_HOME": str(huggingface_cache),
        "UV_CACHE_DIR": str(uv_cache),
        RUN_ID_ENV: spec.run_id,
    }


def build_argv(spec: JobSpec, workspace: Path) -> tuple[list[str], str]:
    """Возвращает итоговый argv и название режима изоляции для лога."""
    if not sandbox.is_enabled():
        return list(spec.argv), "none"

    writable = [cache_dir("huggingface"), cache_dir("uv")]
    return sandbox.wrap(list(spec.argv), workspace=workspace, writable=writable), "bwrap"


# --- уборка процессов ---------------------------------------------------------


def _marked_pids(run_id: str) -> list[int]:
    """Находит процессы этого запуска по маркеру в окружении.

    Маркер наследуется всеми потомками, поэтому находится даже тот, кто ушёл в
    собственную сессию через setsid. В отличие от `pkill -u`, чужой параллельный
    запуск и сам диспетчер при этом не задеваются.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        # macOS: остаётся только уборка по группе процессов.
        return []

    marker = f"{RUN_ID_ENV}={run_id}\0".encode()
    own_pid = os.getpid()
    pids: list[int] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            # Процесс уже исчез или принадлежит другому пользователю.
            continue
        if marker in environ:
            pids.append(pid)
    return pids


def terminate_run(process: subprocess.Popen[bytes] | None, run_id: str) -> None:
    """Гарантированно убирает всё, что запустила команда студента."""
    if process is not None and process.poll() is None:
        _signal_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass

    # Второй слой: потомки, покинувшие группу процессов.
    for attempt in (signal.SIGTERM, signal.SIGKILL):
        pids = _marked_pids(run_id)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, attempt)
            except OSError as error:
                if error.errno != errno.ESRCH:
                    raise
        time.sleep(1.0)


def _signal_group(pid: int, number: int) -> None:
    """Шлёт сигнал всей группе процессов запуска, игнорируя уже умерших."""
    try:
        os.killpg(os.getpgid(pid), number)
    except OSError as error:
        if error.errno not in {errno.ESRCH, errno.EPERM}:
            raise


# --- собственно запуск --------------------------------------------------------


def execute(spec: JobSpec, workspace: Path, devices: str) -> tuple[int, str, float]:
    """Запускает команду и возвращает (код возврата, статус, длительность)."""
    argv, isolation = build_argv(spec, workspace)
    environment = build_environment(spec, workspace, devices)
    log_path = results_dir(workspace, spec.run_id) / "runner.log"

    started = time.monotonic()
    with log_path.open("ab") as log:
        log.write(
            (
                f"=== {utc_now()} run_id={spec.run_id} "
                f"branch={spec.branch}@{spec.commit_sha[:12]} "
                f"gpus={spec.gpus} devices={devices or '-'} isolation={isolation}\n"
                f"=== command: {spec.command}\n"
            ).encode()
        )
        log.flush()

        process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

        try:
            returncode = process.wait(timeout=spec.timeout_seconds)
            status = "done" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            returncode = -signal.SIGKILL
            status = "timeout"
        finally:
            terminate_run(process, spec.run_id)

        duration = time.monotonic() - started
        log.write(
            f"=== {utc_now()} status={status} exit_code={returncode} "
            f"duration={duration:.1f}s\n".encode()
        )

    return returncode, status, duration


def _record_failure(spec: JobSpec, message: str) -> None:
    """Сохраняет причину незапуска и в лог диспетчера, и в runner.log."""
    print(message, file=sys.stderr)

    log_path = results_dir(workspace_path(spec.run_id), spec.run_id) / "runner.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"=== {utc_now()} {message}\n")
    except OSError:
        # Рабочая копия могла не создаться — сообщение уже ушло в лог диспетчера.
        pass


def write_result(spec: JobSpec, status: str, returncode: int, duration: float) -> None:
    """Публикует итог запуска для диспетчера, который его подберёт на следующем тике."""
    write_json_atomic(
        result_path(spec.run_id),
        {
            "run_id": spec.run_id,
            "status": status,
            "exit_code": returncode,
            "duration_seconds": round(duration, 3),
            "finished_at": utc_now(),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="путь спеки в репозитории, например jobs/queue/ivan-math500.yml")
    parser.add_argument(
        "--gpu-indices",
        default="",
        help="номера арендованных карт через запятую (передаёт диспетчер)",
    )
    parser.add_argument(
        "--lease-fds",
        default="",
        help="унаследованные дескрипторы аренды GPU через запятую",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="показать итоговый argv и окружение, ничего не запуская",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_layout()
    repo = repo_dir()

    try:
        spec = load_spec_from_main(PurePosixPath(args.spec), repo)
    except JobSpecError as error:
        print(f"invalid spec {args.spec}: {error}", file=sys.stderr)
        return 2

    devices = args.gpu_indices.strip()
    descriptors = [int(item) for item in args.lease_fds.split(",") if item.strip()]
    # Объект аренды нужен, чтобы дескрипторы не закрылись раньше времени: пока он
    # жив, жива и блокировка.
    lease = adopt_descriptors(descriptors, [int(item) for item in devices.split(",") if item], spec.run_id)

    if args.dry_run:
        workspace = workspace_path(spec.run_id)
        try:
            argv_final, isolation = build_argv(spec, workspace)
        except sandbox.SandboxError as error:
            # Диагностический режим должен показать проблему, а не упасть трассой.
            argv_final, isolation = list(spec.argv), f"BROKEN: {error}"
        print(f"run_id:     {spec.run_id}")
        print(f"branch:     {spec.branch}@{spec.commit_sha}")
        print(f"workspace:  {workspace}")
        print(f"isolation:  {isolation}")
        print(f"devices:    {devices or '-'}")
        print(f"argv:       {argv_final}")
        for key, value in sorted(build_environment(spec, workspace, devices).items()):
            print(f"env:        {key}={value}")
        return 0

    try:
        workspace = prepare_workspace(spec, repo)
        returncode, status, duration = execute(spec, workspace, devices)
    except Exception as error:  # noqa: BLE001 — итог обязан быть записан всегда
        terminate_run(None, spec.run_id)
        write_result(spec, "failed", 1, 0.0)
        # Причина должна попасть в runner.log: именно его хвост уходит в
        # комментарий к PR, и без этого запуск выглядел бы упавшим без объяснений.
        _record_failure(spec, f"run {spec.run_id} failed to start: {error}")
        return 1
    finally:
        lease.release()

    write_result(spec, status, returncode, duration)
    return 0 if status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
