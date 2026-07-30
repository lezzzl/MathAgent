from __future__ import annotations

import ast
from pathlib import Path

from dashboard.artifacts import discover_runs, load_task_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_current_runs_have_expected_pairable_task_counts() -> None:
    runs = discover_runs(PROJECT_ROOT / "results" / "runs")
    assert len(runs) == 2
    run_values = list(runs.values())

    for benchmark, expected in {
        "AIME26": 30,
        "HMMT26": 33,
        "IMOAnswerBench": 400,
    }.items():
        task_sets = []
        for run in run_values:
            records = load_task_records(run.benchmarks[benchmark].path)
            task_sets.append({str(record["task_id"]) for record in records})
        assert len(task_sets[0]) == expected
        assert task_sets[0] == task_sets[1]


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
