"""Каталоги рабочего окружения диспетчера.

Всё лежит внутри домашнего каталога одного непривилегированного пользователя:
ни systemd-юнитов, ни /var/lib, ни sudo. Расположение переопределяется
переменной MATHAGENT_HOME — это же используют тесты.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME = Path.home() / "mathagent"


def mathagent_home() -> Path:
    """Корень рабочего окружения диспетчера."""
    override = os.environ.get("MATHAGENT_HOME")
    return Path(override).expanduser() if override else DEFAULT_HOME


def repo_dir() -> Path:
    """Клон репозитория, из которого диспетчер читает очередь и себя самого."""
    return mathagent_home() / "repo"


def slots_dir() -> Path:
    """Каталог lock-файлов GPU: их количество и есть бюджет карт."""
    return mathagent_home() / "gpu-slots"


def state_dir() -> Path:
    """Состояния запусков, по одному JSON на run_id."""
    return mathagent_home() / "state"


def work_dir() -> Path:
    """Рабочие копии (git worktree) отдельных запусков."""
    return mathagent_home() / "work"


def logs_dir() -> Path:
    """Логи диспетчера и отдельных запусков."""
    return mathagent_home() / "logs"


def sensitive_paths() -> list[Path]:
    """Личные секреты владельца машины, которые песочница прячет от запуска.

    Диспетчер пушит результаты под обычной учётной записью, поэтому именно эти
    файлы дают возможность действовать от её имени. В окружение команды они не
    попадают никогда, а bwrap ещё и убирает их из файловой системы.

    Переопределяется MATHAGENT_SENSITIVE (пути через двоеточие) — это же
    используют тесты.
    """
    override = os.environ.get("MATHAGENT_SENSITIVE")
    if override is not None:
        return [Path(item).expanduser() for item in override.split(":") if item]

    home = Path.home()
    return [home / ".ssh", home / ".config" / "gh", home / ".git-credentials"]


def ensure_layout() -> None:
    """Создаёт недостающие каталоги — вызывается перед любой записью."""
    for directory in (slots_dir(), state_dir(), work_dir(), logs_dir()):
        directory.mkdir(parents=True, exist_ok=True)
