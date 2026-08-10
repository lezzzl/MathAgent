from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Спеки и команды проверяются относительно настоящего git-дерева, поэтому тесты
# собирают маленький репозиторий вместо того, чтобы подменять git заглушками.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def git(repo: Path, *args: str) -> str:
    """Запускает git в тестовом репозитории и возвращает stdout."""
    import os

    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **GIT_ENV},
    )
    return process.stdout.strip()


@pytest.fixture
def student_repo(tmp_path: Path) -> Path:
    """Репозиторий с веткой студента и парой отслеживаемых скриптов."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "benchmarks").mkdir(parents=True)
    (repo / "conf" / "experiments" / "baseline").mkdir(parents=True)
    (repo / "jobs" / "queue").mkdir(parents=True)

    (repo / "scripts" / "benchmarks" / "run_demo.py").write_text("print('demo')\n")
    (repo / "scripts" / "serve.sh").write_text("echo serve\n")
    (repo / "conf" / "experiments" / "baseline" / "parameters.yml").write_text("a: 1\n")
    (repo / ".gitignore").write_text("results/**/*.jsonl\n")

    git(repo.parent, "init", "--quiet", "--initial-branch=main", str(repo))
    git(repo, "add", "-A")
    git(repo, "commit", "--quiet", "-m", "initial")

    # Ветка студента: именно её называет спека.
    git(repo, "branch", "Ilya/demo")
    # Локальные ветки видны как origin/*, чтобы resolve_branch работал как на VM.
    git(repo, "remote", "add", "origin", str(repo))
    git(repo, "fetch", "--quiet", "origin")
    return repo


@pytest.fixture
def slots(tmp_path: Path) -> Path:
    """Каталог с двумя слотами GPU — тот же бюджет, что и на машине."""
    directory = tmp_path / "gpu-slots"
    directory.mkdir()
    (directory / "gpu0.lock").touch()
    (directory / "gpu1.lock").touch()
    return directory
