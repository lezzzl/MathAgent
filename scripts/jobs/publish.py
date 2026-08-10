"""Публикация результатов: коммит в ветку студента и комментарий в PR.

Выполняется только из диспетчера и только после того, как процесс запуска
завершился. Push и комментарий идут под обычной учётной записью владельца
машины: используются уже настроенные ssh-ключ и `gh auth`, отдельный бот не
нужен. К команде студента доступа к ним нет — run_job собирает её окружение с
нуля, а песочница прячет сами файлы.

Каталог results/ лежит в .gitignore, поэтому добавление идёт через `git add -f`.
Ветка студента может уйти вперёд, пока запуск стоял в очереди, — поэтому push
делается после rebase на актуальную вершину и никогда не форсируется.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks.run_artifacts import utc_now

GITHUB_API = "https://api.github.com"

# Предел GitHub на файл — 100 МиБ; берём запас, чтобы push не отклонили целиком.
MAX_FILE_BYTES = 90 * 1024 * 1024

# Сколько строк лога показывать в комментарии к неуспешному запуску.
LOG_TAIL_LINES = 30

REMOTE_PATTERN = re.compile(
    r"(?:git@github\.com:|https://github\.com/)(?P<owner>[^/]+)/(?P<repo>[^/.]+)"
)

GH_CLI = "gh"


class PublishError(RuntimeError):
    """Ошибка публикации: запуск уже завершён, поэтому она не отменяет результат."""


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if check and process.returncode != 0:
        raise PublishError(
            f"{' '.join(args)} failed: {process.stderr.strip() or process.stdout.strip()}"
        )
    return process


def git_environment() -> dict[str, str]:
    """Окружение git для push: обычное окружение диспетчера.

    Ключ и подпись берутся из настроек пользователя — того же, под которым вы
    ходите в этот репозиторий руками.
    """
    return dict(os.environ)


def github_token() -> str | None:
    """Токен для комментария в PR: из окружения или из уже настроенного gh."""
    token = os.environ.get("MATHAGENT_GITHUB_TOKEN")
    if token:
        return token.strip()

    # cron даёт пустое окружение, поэтому спрашиваем gh напрямую, а не полагаемся
    # на GH_TOKEN.
    process = subprocess.run(
        [GH_CLI, "auth", "token"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode == 0 and process.stdout.strip():
        return process.stdout.strip()
    return None


def repository_slug(repo: Path) -> tuple[str, str]:
    """Определяет owner/repo по адресу origin."""
    url = _run(["git", "remote", "get-url", "origin"], cwd=repo).stdout.strip()
    match = REMOTE_PATTERN.search(url)
    if match is None:
        raise PublishError(f"cannot parse GitHub repository from origin url {url!r}")
    return match.group("owner"), match.group("repo")


# --- результаты в ветку студента ----------------------------------------------


def _stage_results(workspace: Path) -> tuple[list[str], list[str]]:
    """Готовит к коммиту всё, что запуск оставил в results/.

    Публикуется весь каталог, а не только results/runs/<run_id>: студент может
    запустить бенчмарк без --run-id, и тогда файлы лягут под сгенерированным
    именем. Слишком большие файлы снимаются со staging — иначе GitHub отклонит
    push целиком, и не опубликуется ничего.
    """
    results = workspace / "results"
    if not results.is_dir():
        return [], []

    _run(["git", "add", "-f", "results"], cwd=workspace)
    staged = _run(
        ["git", "diff", "--cached", "--name-only"], cwd=workspace
    ).stdout.splitlines()

    skipped: list[str] = []
    for name in staged:
        path = workspace / name
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            _run(["git", "restore", "--staged", name], cwd=workspace)
            skipped.append(name)

    published = [name for name in staged if name not in skipped]
    return published, skipped


def push_results(
    workspace: Path,
    *,
    branch: str,
    run_id: str,
    status: str,
    exit_code: int,
) -> dict[str, Any]:
    """Коммитит результаты и отправляет их в ветку студента."""
    published, skipped = _stage_results(workspace)
    if not published:
        return {"pushed": False, "files": [], "skipped": skipped}

    environment = git_environment()
    _run(
        ["git", "commit", "-m", f"results({run_id}): {status}, exit {exit_code}"],
        cwd=workspace,
        env=environment,
    )

    # Одна повторная попытка покрывает обычный случай: студент запушил в ветку,
    # пока запуск стоял в очереди или считался.
    last_error = ""
    for _attempt in range(2):
        _run(["git", "fetch", "origin", branch], cwd=workspace, env=environment)
        rebase = _run(
            ["git", "rebase", f"origin/{branch}"], cwd=workspace, env=environment, check=False
        )
        if rebase.returncode != 0:
            _run(["git", "rebase", "--abort"], cwd=workspace, check=False)
            last_error = rebase.stderr.strip()
            continue

        push = _run(
            ["git", "push", "origin", f"HEAD:{branch}"],
            cwd=workspace,
            env=environment,
            check=False,
        )
        if push.returncode == 0:
            return {"pushed": True, "files": published, "skipped": skipped}
        last_error = push.stderr.strip()

    raise PublishError(f"cannot push results to {branch}: {last_error}")


# --- комментарий в PR ---------------------------------------------------------


def find_pull_request(repo: Path, spec_path: str) -> int | None:
    """Находит PR, которым спека попала в main."""
    commit = _run(
        [
            "git", "log", "--diff-filter=A", "--format=%H", "-1",
            "origin/main", "--", spec_path,
        ],
        cwd=repo,
    ).stdout.strip()
    if not commit:
        return None

    owner, name = repository_slug(repo)
    payload = github_request("GET", f"/repos/{owner}/{name}/commits/{commit}/pulls")
    if not payload:
        return None
    return int(payload[0]["number"])


def github_request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Минимальный клиент GitHub API на urllib: лишняя зависимость тут не нужна."""
    token = github_token()
    if token is None:
        raise PublishError(
            "no GitHub token — run `gh auth login` under the dispatcher user "
            "(see jobs/RUNNER.md)"
        )

    request = urllib.request.Request(
        f"{GITHUB_API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "mathagent-jobs",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode() or "null")
    except urllib.error.HTTPError as error:
        raise PublishError(
            f"GitHub API {method} {path} failed: {error.code} {error.read().decode()[:200]}"
        ) from error
    except urllib.error.URLError as error:
        raise PublishError(f"GitHub API {method} {path} unreachable: {error.reason}") from error


def _log_tail(workspace: Path, run_id: str) -> str:
    """Последние строки runner.log — по ним студент чаще всего и чинит запуск."""
    log_path = workspace / "results" / "runs" / run_id / "runner.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-LOG_TAIL_LINES:])


def build_comment(state: dict[str, Any], workspace: Path, push: dict[str, Any]) -> str:
    """Собирает текст комментария к PR."""
    status = state.get("status", "unknown")
    icon = {"done": "✅", "failed": "❌", "timeout": "⏱", "orphaned": "💥"}.get(status, "ℹ️")
    run_id = state["run_id"]

    lines = [
        f"{icon} **{run_id}** — `{status}` (exit {state.get('exit_code')})",
        "",
        f"- ветка: `{state['branch']}` @ `{str(state['commit_sha'])[:12]}`",
        f"- команда: `{state['command']}`",
        f"- GPU: {state.get('gpus', 0)}"
        + (f" (карты {state['devices']})" if state.get("devices") else ""),
        f"- ожидание в очереди: {state.get('queue_wait_seconds', 0):.0f} с",
        f"- длительность: {state.get('duration_seconds', 0):.0f} с",
    ]

    if push.get("pushed"):
        lines.append(f"- результаты запушены в `{state['branch']}` ({len(push['files'])} файл(ов))")
    else:
        lines.append("- результаты не публиковались: запуск ничего не записал в `results/`")
    if push.get("skipped"):
        skipped = ", ".join(f"`{name}`" for name in push["skipped"])
        lines.append(f"- пропущены слишком большие файлы: {skipped}")

    if status != "done":
        tail = _log_tail(workspace, run_id)
        if tail:
            lines += ["", "<details><summary>последние строки runner.log</summary>", "",
                      "```", tail, "```", "", "</details>"]

    return "\n".join(lines)


def comment_on_pull_request(repo: Path, pull_number: int, body: str) -> None:
    owner, name = repository_slug(repo)
    github_request(
        "POST",
        f"/repos/{owner}/{name}/issues/{pull_number}/comments",
        {"body": body},
    )


def publish(repo: Path, workspace: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Публикует результаты и статус, возвращая сведения для state-файла.

    Ошибки не пробрасываются наружу: запуск уже завершён, и падение публикации не
    должно мешать диспетчеру освободить очередь. Причина сохраняется в state.
    """
    report: dict[str, Any] = {"published_at": None, "publish_error": None}

    try:
        push = push_results(
            workspace,
            branch=state["branch"],
            run_id=state["run_id"],
            status=state.get("status", "unknown"),
            exit_code=state.get("exit_code", -1),
        )
        report["pushed_files"] = push["files"]
        report["skipped_files"] = push["skipped"]
    except PublishError as error:
        push = {"pushed": False, "files": [], "skipped": []}
        report["publish_error"] = str(error)

    try:
        pull_number = state.get("pr_number") or find_pull_request(repo, state["spec"])
        if pull_number is not None:
            comment_on_pull_request(repo, pull_number, build_comment(state, workspace, push))
            report["pr_number"] = pull_number
    except PublishError as error:
        previous = report["publish_error"]
        report["publish_error"] = f"{previous}; {error}" if previous else str(error)

    report["published_at"] = utc_now()
    return report
