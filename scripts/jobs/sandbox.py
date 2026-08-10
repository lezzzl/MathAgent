"""Песочница для команды студента.

Диспетчер работает под обычной учётной записью владельца машины и под ней же
пушит результаты. Без изоляции команда студента прочитала бы его ssh-ключ и
токен gh, то есть смогла бы действовать от его имени.

bubblewrap закрывает это без привилегий: пространство имён пользователя делает
корень доступным только на чтение, оставляет запись в рабочем каталоге запуска
и подменяет каталоги с секретами пустым tmpfs.

Песочница включена по умолчанию. Если bwrap на машине не работает, запуск не
начинается: тихо выполнить команду без изоляции хуже, чем не выполнить вовсе.
Выключается только явно — файлом `sandbox.disabled` или MATHAGENT_SANDBOX=0.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.jobs.paths import mathagent_home, sensitive_paths

BWRAP = "bwrap"

# Наличие файла выключает песочницу; переменная окружения перекрывает его в обе
# стороны, чтобы режим можно было проверить одной командой.
DISABLE_FLAG_NAME = "sandbox.disabled"
SANDBOX_ENV = "MATHAGENT_SANDBOX"


class SandboxError(RuntimeError):
    """Песочница включена, но не работает — запускать команду нельзя."""


def is_enabled() -> bool:
    """Включена ли песочница. По умолчанию — да."""
    override = os.environ.get(SANDBOX_ENV)
    if override is not None:
        return override.strip() not in {"", "0", "false", "no"}
    return not (mathagent_home() / DISABLE_FLAG_NAME).exists()


def probe() -> tuple[bool, str]:
    """Проверяет, работает ли bubblewrap на этой машине.

    Возвращает (успех, пояснение) — пояснение стоит записать в RUNNER.md, чтобы
    причина выключенной песочницы не выяснялась заново через полгода.
    """
    if shutil.which(BWRAP) is None:
        return False, "bubblewrap (bwrap) is not installed: apt install bubblewrap"

    process = subprocess.run(
        [BWRAP, "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc", "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return False, (
            "bwrap cannot create a user namespace: "
            f"{process.stderr.strip() or 'unknown error'}"
        )
    return True, "bubblewrap works"


def wrap(
    argv: list[str],
    *,
    workspace: Path,
    writable: list[Path],
) -> list[str]:
    """Оборачивает команду в bwrap: корень только на чтение, запись — в workspace.

    Порядок аргументов важен: более поздние привязки перекрывают ранние, поэтому
    `--ro-bind / /` идёт первым, а рабочие каталоги — после него.
    """
    working, explanation = probe()
    if not working:
        raise SandboxError(
            f"{explanation}. Fix it, or turn the sandbox off deliberately: "
            f"touch {mathagent_home() / DISABLE_FLAG_NAME}"
        )

    command = [
        BWRAP,
        # Всё дерево видно, но неизменяемо.
        "--ro-bind", "/", "/",
        # Доступ к картам: /dev/nvidia* нужен вычислениям.
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
    ]

    # Пустой tmpfs вместо ssh-ключа и токена gh: под ними диспетчер пушит
    # результаты, поэтому команде студента их видеть незачем.
    for path in sensitive_paths():
        if path.exists():
            command += ["--tmpfs", str(path)]

    # Запись разрешена только в рабочий каталог запуска и общие кэши.
    command += ["--bind", str(workspace), str(workspace)]
    for path in writable:
        command += ["--bind", str(path), str(path)]

    command += [
        # Потомки не переживут смерть запуска, даже если сделают setsid.
        "--die-with-parent",
        "--new-session",
        "--",
        *argv,
    ]
    return command
