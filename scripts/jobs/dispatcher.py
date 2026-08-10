"""Тик диспетчера: подобрать завершившиеся запуски и раздать свободные GPU.

Запускается из cron раз в минуту под `flock -n`, поэтому два тика не пересекаются.
Тик обязан быть коротким: запуски он отсоединяет через setsid и сразу выходит —
иначе часовой бенчмарк заблокировал бы всю очередь.

Порядок очереди — FIFO по времени появления спеки в main. Если первой в очереди
не хватает карт, тик останавливается, а не пропускает её вперёд: без этого
поток однокарточных запусков бесконечно откладывал бы двухкарточный.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks.run_artifacts import (
    read_json,
    utc_now,
    write_json_atomic,
)
from scripts.jobs import publish as publish_module
from scripts.jobs.gpu_lease import GpuLease, GpuLeaseError, slot_holders
from scripts.jobs.job_spec import (
    QUEUE_DIR,
    JobSpec,
    JobSpecError,
    git_output,
    load_spec_text,
)
from scripts.jobs.paths import (
    ensure_layout,
    logs_dir,
    repo_dir,
    state_dir,
    work_dir,
)
from scripts.jobs.run_job import result_path, workspace_path

# Лог диспетчера обрезается им самим: системного logrotate у нас нет.
MAX_LOG_BYTES = 10 * 1024 * 1024

RUNNING_STATUSES = frozenset({"running"})


def log(message: str) -> None:
    """Пишет строку тика в stdout — cron перенаправляет его в dispatcher.log."""
    print(f"{utc_now()} {message}", flush=True)


def state_path(run_id: str) -> Path:
    return state_dir() / f"{run_id}.json"


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def process_alive(pid: int) -> bool:
    """Жив ли процесс запуска. Сигнал 0 только проверяет существование."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Чужой процесс с тем же pid — считаем, что наш уже умер.
        return False
    return True


# --- подбор завершившихся запусков --------------------------------------------


def reap(repo: Path) -> None:
    """Находит завершившиеся запуски, публикует их и освобождает рабочие копии."""
    for path in sorted(state_dir().glob("*.json")):
        if path.name.endswith(".result.json"):
            continue
        state = read_state(path)
        if state is None or state.get("status") not in RUNNING_STATUSES:
            continue
        if process_alive(int(state["pid"])):
            continue

        run_id = state["run_id"]
        result = read_state(result_path(run_id))
        if result is None:
            # Процесс исчез, не записав итог: убит извне или упал целиком.
            state.update(status="orphaned", exit_code=None, duration_seconds=0.0)
        else:
            state.update(
                status=result["status"],
                exit_code=result["exit_code"],
                duration_seconds=result["duration_seconds"],
                finished_at=result["finished_at"],
            )
        state.setdefault("finished_at", utc_now())

        log(f"reap run_id={run_id} status={state['status']} exit={state.get('exit_code')}")
        workspace = workspace_path(run_id)
        state.update(publish_module.publish(repo, workspace, state))
        if state.get("publish_error"):
            log(f"publish_error run_id={run_id} {state['publish_error']}")

        write_json_atomic(path, state)
        result_path(run_id).unlink(missing_ok=True)
        remove_workspace(repo, workspace)


def remove_workspace(repo: Path, workspace: Path) -> None:
    """Убирает рабочую копию: результаты уже опубликованы."""
    if not workspace.exists():
        return
    process = subprocess.run(
        ["git", "worktree", "remove", "--force", str(workspace)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        log(f"worktree_remove_failed path={workspace} {process.stderr.strip()}")


# --- очередь ------------------------------------------------------------------


def queued_spec_paths(repo: Path) -> list[tuple[PurePosixPath, int]]:
    """Спеки из main вместе со временем их появления, в порядке FIFO."""
    listing = git_output(
        repo, "ls-tree", "--name-only", "origin/main", f"{QUEUE_DIR}/"
    )
    paths = [
        PurePosixPath(line)
        for line in listing.splitlines()
        if line.endswith(".yml")
    ]

    def added_at(path: PurePosixPath) -> int:
        """Момент, когда спека попала в main, — он же место в очереди."""
        stamp = git_output(
            repo, "log", "--diff-filter=A", "--format=%ct", "-1", "origin/main", "--", str(path)
        )
        return int(stamp) if stamp else 0

    return sorted(
        ((path, added_at(path)) for path in paths),
        key=lambda item: (item[1], str(item[0])),
    )


def pending_specs(repo: Path) -> list[tuple[PurePosixPath, int]]:
    """Спеки без состояния — то есть ещё не запускавшиеся — с моментом попадания в очередь."""
    return [
        (path, added_at)
        for path, added_at in queued_spec_paths(repo)
        if not state_path(path.stem).exists()
    ]


def mark_invalid(repo: Path, spec_path: PurePosixPath, error: str) -> None:
    """Фиксирует непроходную спеку, чтобы не пытаться запускать её каждую минуту."""
    run_id = spec_path.stem
    state = {
        "run_id": run_id,
        "spec": str(spec_path),
        "status": "invalid",
        "error": error,
        "queued_at": utc_now(),
        "finished_at": utc_now(),
    }
    write_json_atomic(state_path(run_id), state)
    log(f"invalid spec={spec_path} {error}")


# --- запуск -------------------------------------------------------------------


def spawn(repo: Path, spec: JobSpec, lease: GpuLease, queued_at: int) -> None:
    """Отсоединяет процесс запуска и записывает состояние.

    Дескрипторы аренды передаются потомку: блокировка живёт, пока открыт хотя бы
    один из них, поэтому после detach() карты остаются за запуском, а не за
    завершающимся тиком.
    """
    logs_dir().mkdir(parents=True, exist_ok=True)
    job_log = logs_dir() / f"{spec.run_id}.log"

    descriptors = lease.descriptors
    command = [
        sys.executable,
        str(repo / "scripts" / "jobs" / "run_job.py"),
        str(spec.path),
        "--gpu-indices",
        lease.cuda_visible_devices,
        "--lease-fds",
        ",".join(str(descriptor) for descriptor in descriptors),
    ]

    with job_log.open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=descriptors,
        )

    # Свои копии дескрипторов закрываем: аренду теперь держит потомок.
    lease.detach()

    write_json_atomic(
        state_path(spec.run_id),
        {
            "run_id": spec.run_id,
            "spec": str(spec.path),
            "student": spec.student,
            "branch": spec.branch,
            "commit_sha": spec.commit_sha,
            "command": spec.command,
            "gpus": spec.gpus,
            "devices": lease.cuda_visible_devices,
            "timeout_minutes": spec.timeout_minutes,
            "status": "running",
            "pid": process.pid,
            "queued_at": datetime.fromtimestamp(queued_at, tz=timezone.utc).isoformat(),
            "started_at": utc_now(),
            "queue_wait_seconds": max(0.0, time.time() - queued_at),
        },
    )
    log(
        f"start run_id={spec.run_id} pid={process.pid} gpus={spec.gpus} "
        f"devices={lease.cuda_visible_devices or '-'} branch={spec.branch}"
    )


def dispatch(repo: Path) -> None:
    """Раздаёт свободные карты головным спекам очереди."""
    for spec_path, queued_at in pending_specs(repo):
        try:
            text = git_output(repo, "show", f"origin/main:{spec_path}")
            spec = load_spec_text(text, spec_path=spec_path, repo=repo)
        except JobSpecError as error:
            mark_invalid(repo, spec_path, str(error))
            continue

        lease = GpuLease(spec.gpus, run_id=spec.run_id)
        try:
            acquired = lease.acquire()
        except GpuLeaseError as error:
            mark_invalid(repo, spec_path, str(error))
            continue

        if not acquired:
            holders = {index: run for index, run in slot_holders().items() if run}
            busy = ", ".join(f"gpu{index}={run}" for index, run in sorted(holders.items()))
            log(
                f"waiting run_id={spec.run_id} needs={spec.gpus} "
                f"busy=[{busy or 'none'}] — queue held"
            )
            return

        try:
            spawn(repo, spec, lease, queued_at)
        except Exception:
            lease.release()
            raise


# --- тик ----------------------------------------------------------------------


def truncate_log() -> None:
    """Обрезает разросшийся лог диспетчера — системного logrotate у нас нет."""
    path = logs_dir() / "dispatcher.log"
    if path.exists() and path.stat().st_size > MAX_LOG_BYTES:
        path.write_text("", encoding="utf-8")


def tick() -> int:
    ensure_layout()
    work_dir().mkdir(parents=True, exist_ok=True)
    repo = repo_dir()

    git_output(repo, "fetch", "--prune", "origin")
    reap(repo)
    dispatch(repo)
    truncate_log()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        action="store_true",
        help="показать очередь и занятые карты, ничего не запуская",
    )
    return parser.parse_args(argv)


def show_status() -> int:
    """Короткая сводка для человека: что выполняется и что ждёт."""
    ensure_layout()
    print("GPU:")
    for index, holder in sorted(slot_holders().items()):
        print(f"  gpu{index}: {holder or 'free'}")

    print("Запуски:")
    for path in sorted(state_dir().glob("*.json")):
        if path.name.endswith(".result.json"):
            continue
        state = read_state(path)
        if state is None:
            continue
        print(
            f"  {state['run_id']:<32} {state.get('status', '?'):<10} "
            f"exit={state.get('exit_code')} {state.get('branch', '')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        return show_status()
    return tick()


if __name__ == "__main__":
    raise SystemExit(main())
