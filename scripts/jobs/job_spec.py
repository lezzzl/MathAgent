"""Разбор и валидация job-спеки студента.

Модуль намеренно один на две стороны: его вызывает и PR-проверка на GitHub,
и диспетчер на VM. Если бы проверок было две, они разошлись бы, и на машину
попала бы команда, которую PR-проверка считала безопасной.

Спека описывает *что* запустить, а не *как*: ветка студента, команда и число
GPU. Команда разбирается без участия shell, поэтому все метасимволы отклоняются
до запуска, а сам скрипт обязан существовать в репозитории на указанной ветке.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.run_artifacts import validate_run_id

# Каталог, в котором студенты публикуют спеки. Всё остальное в PR со спекой
# менять не следует — это проверяется на ревью (см. jobs/RUNNER.md).
QUEUE_DIR = PurePosixPath("jobs/queue")

# Верхние границы ресурсов. Число GPU ограничено бюджетом машины, таймаут —
# чтобы забытый запуск не занимал карту до конца смены.
MAX_GPUS = 2
MAX_TIMEOUT_MINUTES = 120

# Спека содержит ровно эти поля: лишние ключи почти всегда опечатка, а молча
# проигнорированная опечатка означает запуск не с теми параметрами.
REQUIRED_FIELDS = frozenset({"branch", "command", "gpus", "timeout_minutes"})

EXPERIMENT_PATTERN = re.compile(r"^experiments/[A-Za-z0-9._-]+$")

# Ветки, из которых разрешено запускать. Список правится руками при появлении
# нового человека на курсе — этого достаточно, отдельной проверки не нужно.
BRANCH_PREFIXES = ("Ilya/", "Stas/", "pavel/", "Nikita/")

# Общая ветка курса: код, не привязанный к конкретному студенту.
SHARED_BRANCHES = frozenset({"main"})

BRANCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

# Ни один из этих символов не имеет смысла без shell: команда запускается через
# execve. Их присутствие означает либо попытку инъекции, либо непонимание того,
# как выполняется команда — в обоих случаях лучше отказать явно.
FORBIDDEN_COMMAND_CHARS = ";|&$`()<>\n\r"

# Интерпретаторы, с которых может начинаться команда. Всё остальное (curl, make,
# произвольный бинарник) отклоняется.
INTERPRETERS = frozenset({"python", "python3", "bash", "sh", "uv"})


class JobSpecError(ValueError):
    """Ошибка спеки, текст которой показывается студенту в PR."""


@dataclass(frozen=True)
class JobSpec:
    """Проверенная спека: все поля уже пригодны для запуска без доп. проверок."""

    path: PurePosixPath
    branch: str
    command: str
    gpus: int
    timeout_minutes: int
    commit_sha: str
    argv: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        """Идентификатор запуска: имя файла спеки без расширения.

        Оно же имя каталога результатов, поэтому повторно использовать имя файла
        нельзя — диспетчер узнаёт по нему уже отработавший запуск.
        """
        return self.path.stem

    @property
    def student(self) -> str:
        """Владелец запуска — по префиксу ветки; нужен только для сообщений."""
        return self.branch.split("/", 1)[0]

    @property
    def timeout_seconds(self) -> float:
        return float(self.timeout_minutes) * 60.0


def git_output(repo: Path, *args: str) -> str:
    """Выполняет git и возвращает stdout, превращая ошибку в JobSpecError."""
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise JobSpecError(
            f"git {' '.join(args)} failed: {process.stderr.strip() or 'unknown error'}"
        )
    return process.stdout.strip()


def git_succeeds(repo: Path, *args: str) -> bool:
    """Проверяет успешность git-команды, когда важен только код возврата."""
    process = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.returncode == 0


def resolve_branch(repo: Path, branch: str) -> str:
    """Проверяет имя ветки и превращает его в конкретный commit sha.

    Ветка фиксируется в sha сразу: запуск и публикация результатов должны
    относиться к одному коммиту, даже если студент запушит новый во время
    ожидания в очереди.
    """
    if branch not in SHARED_BRANCHES:
        if not BRANCH_PATTERN.fullmatch(branch) or ".." in branch:
            raise JobSpecError(f"branch {branch!r} is not a valid branch name")
        if not branch.startswith(BRANCH_PREFIXES):
            allowed = ", ".join(f"{prefix}…" for prefix in BRANCH_PREFIXES)
            raise JobSpecError(
                f"branch {branch!r} must start with one of: {allowed}"
            )

    # Сначала ищем remote-ветку: локальных копий чужих веток на VM нет — есть
    # только origin/*.
    for ref in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
        if git_succeeds(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"):
            return git_output(repo, "rev-parse", f"{ref}^{{commit}}")

    raise JobSpecError(
        f"branch {branch!r} not found — push it before opening the pull request"
    )


def _check_script_path(raw_path: str) -> PurePosixPath:
    """Отсеивает пути, выходящие за пределы репозитория, до обращения к git."""
    if raw_path.startswith("-"):
        raise JobSpecError(
            f"expected a script path, got the option {raw_path!r}; "
            "flags like -m or -c are not allowed"
        )

    path = PurePosixPath(raw_path)
    if path.is_absolute():
        raise JobSpecError(f"script path {raw_path!r} must be relative to the repo root")
    if ".." in path.parts:
        raise JobSpecError(f"script path {raw_path!r} must not contain '..'")
    return path


def _require_tracked(repo: Path, commit_sha: str, path: PurePosixPath) -> None:
    """Требует, чтобы путь существовал в дереве коммита.

    Это и есть проверка «скрипт из репозитория MathAgent»: файл, лежащий рядом
    на диске, сгенерированный на лету или попавший в .gitignore, не пройдёт.
    """
    if not git_succeeds(repo, "cat-file", "-e", f"{commit_sha}:{path}"):
        raise JobSpecError(
            f"{path} is not tracked in the repository at the requested branch — "
            "commit and push it first"
        )


def _check_suffix(path: PurePosixPath, expected: str, interpreter: str) -> None:
    if path.suffix != expected:
        raise JobSpecError(
            f"{interpreter} expects a {expected} script, got {path}"
        )


def parse_command(
    command: str,
    *,
    repo: Path,
    commit_sha: str,
) -> list[str]:
    """Разбирает команду студента в argv и проверяет, что она запускает скрипт репозитория.

    Возвращаемый argv передаётся в subprocess без shell, поэтому кавычки и
    пробелы внутри аргументов разбираются здесь один раз и больше не
    интерпретируются.
    """
    if not command.strip():
        raise JobSpecError("command must not be empty")

    forbidden = sorted({char for char in command if char in FORBIDDEN_COMMAND_CHARS})
    if forbidden:
        readable = " ".join(repr(char) for char in forbidden)
        raise JobSpecError(
            f"command contains shell metacharacters ({readable}); the command is "
            "executed directly, without a shell, so they cannot work — put the logic "
            "into a script in the repository instead"
        )

    try:
        argv = shlex.split(command, posix=True)
    except ValueError as error:
        raise JobSpecError(f"cannot parse command: {error}") from error

    if not argv:
        raise JobSpecError("command must not be empty")

    interpreter = argv[0]
    if interpreter not in INTERPRETERS:
        allowed = ", ".join(sorted(INTERPRETERS))
        raise JobSpecError(
            f"command must start with one of: {allowed}; got {interpreter!r}"
        )

    if interpreter in {"python", "python3", "bash", "sh"}:
        if len(argv) < 2:
            raise JobSpecError(f"{interpreter} requires a script path")
        script = _check_script_path(argv[1])
        _check_suffix(script, ".py" if interpreter.startswith("python") else ".sh", interpreter)
        _require_tracked(repo, commit_sha, script)
        return argv

    # Дальше только uv: две разрешённые формы, соответствующие тому, как проект
    # запускается вручную (см. README).
    if argv[1:3] == ["run", "python"]:
        if len(argv) < 4:
            raise JobSpecError("'uv run python' requires a script path")
        script = _check_script_path(argv[3])
        _check_suffix(script, ".py", "uv run python")
        _require_tracked(repo, commit_sha, script)
        return argv

    if argv[1:4] == ["run", "kedro", "run"]:
        _check_kedro_environment(argv, repo=repo, commit_sha=commit_sha)
        return argv

    raise JobSpecError(
        "uv is allowed only as 'uv run python <script.py> ...' or "
        "'uv run kedro run --env experiments/<name> ...'"
    )


def _check_kedro_environment(argv: list[str], *, repo: Path, commit_sha: str) -> None:
    """Требует у kedro-запуска существующее окружение из conf/experiments."""
    try:
        env_index = argv.index("--env") + 1
    except ValueError:
        raise JobSpecError(
            "'uv run kedro run' requires --env experiments/<name>"
        ) from None

    if env_index >= len(argv):
        raise JobSpecError("--env requires a value")

    environment = argv[env_index]
    if not EXPERIMENT_PATTERN.fullmatch(environment):
        raise JobSpecError(
            f"--env must look like 'experiments/<name>', got {environment!r}"
        )

    _require_tracked(repo, commit_sha, PurePosixPath("conf") / environment / "parameters.yml")


def _require_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    """Приводит числовое поле к int, отвергая bool и значения вне диапазона."""
    # bool — подкласс int, а 'gpus: true' почти наверняка ошибка.
    if isinstance(value, bool) or not isinstance(value, int):
        raise JobSpecError(f"{name} must be an integer, got {value!r}")
    if not minimum <= value <= maximum:
        raise JobSpecError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _require_str(document: dict[str, Any], name: str) -> str:
    value = document[name]
    if not isinstance(value, str):
        raise JobSpecError(f"{name} must be a string, got {value!r}")
    return value


def parse_spec_document(
    document: Any,
    *,
    spec_path: PurePosixPath,
    repo: Path,
) -> JobSpec:
    """Проверяет уже разобранный YAML и возвращает готовую к запуску спеку."""
    if not isinstance(document, dict):
        raise JobSpecError("spec must be a YAML mapping")

    keys = set(document)
    missing = REQUIRED_FIELDS - keys
    if missing:
        raise JobSpecError(f"missing required fields: {', '.join(sorted(missing))}")
    unknown = keys - REQUIRED_FIELDS
    if unknown:
        raise JobSpecError(
            f"unknown fields: {', '.join(sorted(unknown))}; "
            f"allowed fields are {', '.join(sorted(REQUIRED_FIELDS))}"
        )

    # run_id — имя файла: оно же станет каталогом результатов, поэтому проверяем
    # его теми же правилами, что и --run-id у бенчмарков.
    if spec_path.suffix != ".yml":
        raise JobSpecError(f"spec file must end with .yml, got {spec_path.name}")
    try:
        validate_run_id(spec_path.stem)
    except ValueError as error:
        raise JobSpecError(f"file name is not a valid run_id: {error}") from error

    gpus = _require_int(document["gpus"], "gpus", minimum=0, maximum=MAX_GPUS)
    timeout_minutes = _require_int(
        document["timeout_minutes"],
        "timeout_minutes",
        minimum=1,
        maximum=MAX_TIMEOUT_MINUTES,
    )

    branch = _require_str(document, "branch")
    commit_sha = resolve_branch(repo, branch)

    command = _require_str(document, "command")
    argv = parse_command(command, repo=repo, commit_sha=commit_sha)

    return JobSpec(
        path=spec_path,
        branch=branch,
        command=command,
        gpus=gpus,
        timeout_minutes=timeout_minutes,
        commit_sha=commit_sha,
        argv=argv,
    )


def load_spec_text(
    text: str,
    *,
    spec_path: PurePosixPath,
    repo: Path,
) -> JobSpec:
    """Разбирает спеку из строки — так её читает диспетчер через `git show`."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise JobSpecError(f"invalid YAML: {error}") from error
    return parse_spec_document(document, spec_path=spec_path, repo=repo)


def load_spec(path: Path, *, repo: Path) -> JobSpec:
    """Разбирает спеку из файла рабочей копии — так её читает PR-проверка."""
    spec_path = PurePosixPath(path.as_posix())
    try:
        relative = spec_path.relative_to(PurePosixPath(repo.resolve().as_posix()))
    except ValueError:
        relative = spec_path
    return load_spec_text(
        path.read_text(encoding="utf-8"),
        spec_path=relative,
        repo=repo,
    )
