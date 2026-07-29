from __future__ import annotations

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.artifacts import discover_runs, load_task_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_current_runs_have_expected_pairable_task_counts() -> None:
    runs = discover_runs(PROJECT_ROOT / "results" / "runs")
    assert len(runs) >= 2

    for benchmark, expected in {
        "AIME26": 30,
        "HMMT26": 33,
        "IMOAnswerBench": 400,
    }.items():
        task_sets = []
        for run in runs.values():
            if benchmark not in run.benchmarks:
                continue
            records = load_task_records(run.benchmarks[benchmark].path)
            task_sets.append({str(record["task_id"]) for record in records})
        assert len(task_sets) >= 2
        assert all(len(task_set) == expected for task_set in task_sets)
        assert all(task_set == task_sets[0] for task_set in task_sets[1:])


def test_dashboard_does_not_import_repository_application_code() -> None:
    dashboard_root = PROJECT_ROOT / "src" / "dashboard"
    forbidden = {"mathagent", "scripts"}
    violations: list[str] = []
    for path in dashboard_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".", 1)[0]}
            else:
                continue
            if roots & forbidden:
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations


def test_dashboard_has_one_incremental_update_button() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "dashboard" / "app.py")).run(
        timeout=20
    )

    assert not app.exception
    assert [button.label for button in app.button] == ["Update comparison table"]
    assert all(selectbox.label != "Pre-scored run" for selectbox in app.selectbox)


def test_dashboard_uses_storage_path_defaults(monkeypatch) -> None:
    monkeypatch.delenv("MATHAGENT_RESULTS_DIR", raising=False)
    monkeypatch.delenv("MATHAGENT_COMPARISONS_PATH", raising=False)

    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "dashboard" / "app.py")).run(
        timeout=20
    )

    assert not app.exception
    fields = {field.label: field.value for field in app.text_input}
    assert fields == {
        "Runs directory": "/mnt/storage-1/MathAgent/results/runs",
        "Comparison table": (
            "/mnt/storage-1/MathAgent/results/comparison/table.parquet"
        ),
    }


def test_dashboard_path_environment_overrides(monkeypatch) -> None:
    runs_path = "/custom/results/runs"
    comparisons_path = "/custom/results/comparisons.parquet"
    monkeypatch.setenv("MATHAGENT_RESULTS_DIR", runs_path)
    monkeypatch.setenv("MATHAGENT_COMPARISONS_PATH", comparisons_path)

    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "dashboard" / "app.py")).run(
        timeout=20
    )

    assert not app.exception
    fields = {field.label: field.value for field in app.text_input}
    assert fields == {
        "Runs directory": runs_path,
        "Comparison table": comparisons_path,
    }
