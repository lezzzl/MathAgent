from __future__ import annotations

import json
from pathlib import Path

import pytest
from kedro.config import OmegaConfigLoader

from mathagent.pipeline_registry import register_pipelines
from mathagent.pipelines.experiment.nodes import (
    evaluate_selected_run,
    run_agentic_loop,
    update_comparison_table,
)
from mathagent.settings import CONFIG_LOADER_ARGS
from scripts.evaluate_experiment import (
    build_parser as build_evaluate_parser,
    main as evaluate_main,
)
from scripts.run_experiment import (
    build_parser as build_run_parser,
    load_experiment_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_scripts(root: Path) -> tuple[Path, Path]:
    runner = root / "fake_runner.py"
    runner.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.output, "w", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "benchmark_name": "Bench",
        "task_id": "1",
        "solution": "1",
        "ground_truth": "1",
    }) + "\\n")
""".strip(),
        encoding="utf-8",
    )
    evaluator = root / "fake_evaluator.py"
    evaluator.write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.input, encoding="utf-8") as source:
    records = [json.loads(line) for line in source if line.strip()]
with open(args.output, "w", encoding="utf-8") as target:
    for record in records:
        record["is_correct"] = True
        target.write(json.dumps(record) + "\\n")
""".strip(),
        encoding="utf-8",
    )
    return runner, evaluator


def _config(
    root: Path,
    runner: Path,
    evaluator: Path,
    run_id: str,
) -> dict[str, object]:
    return {
        "project_root": str(root),
        "runs_dir": "results/runs",
        "comparisons_path": "results/comparisons.parquet",
        "run_id": run_id,
        "agent": {
            "args": [],
            "benchmarks": [
                {
                    "name": "Bench",
                    "script": str(runner),
                    "output": "bench.jsonl",
                }
            ],
        },
        "manifest": {"model": "fake-model", "pipeline": "test"},
        "evaluation": {
            "default": {
                "script": str(evaluator),
                "output_suffix": "_verified",
                "args": [],
            }
        },
        "comparison": {"n_resamples": 10, "seed": 42},
    }


def test_workflow_stages_create_score_and_comparison_artifacts(tmp_path: Path) -> None:
    runner, evaluator = _write_fake_scripts(tmp_path)

    first_config = _config(tmp_path, runner, evaluator, "run-a")
    first = run_agentic_loop(first_config)
    first_evaluation = evaluate_selected_run(first_config)

    raw = Path(first["run_directory"]) / "bench.jsonl"
    verified = Path(first["run_directory"]) / "bench_verified.jsonl"
    assert raw.is_file()
    assert verified.is_file()
    assert "is_correct" not in json.loads(raw.read_text(encoding="utf-8"))
    assert json.loads(verified.read_text(encoding="utf-8"))["is_correct"] is True
    assert first_evaluation["evaluated_benchmarks"] == ["Bench"]

    second_config = _config(tmp_path, runner, evaluator, "run-b")
    run_agentic_loop(second_config)
    evaluate_selected_run(second_config)
    update = update_comparison_table(second_config)

    assert Path(update["comparison_path"]).is_file()
    assert update["added_runs"] == ["run-a", "run-b"]
    assert update["comparison_rows_added"] == 1


def test_evaluation_is_idempotent(tmp_path: Path) -> None:
    runner, evaluator = _write_fake_scripts(tmp_path)
    config = _config(tmp_path, runner, evaluator, "run-a")
    run_agentic_loop(config)
    evaluate_selected_run(config)

    repeated = evaluate_selected_run(config)

    assert repeated["evaluated_benchmarks"] == []
    assert repeated["skipped_benchmarks"] == ["Bench"]


def test_registry_exposes_modular_stage_combinations() -> None:
    pipelines = register_pipelines()

    assert {
        "__default__",
        "experiment",
        "agent",
        "evaluate",
        "compare",
        "evaluate_compare",
    } == set(pipelines)
    assert [node.name for node in pipelines["evaluate_compare"].nodes] == [
        "evaluate_results",
        "update_comparison_table",
    ]


def test_full_run_script_requires_config() -> None:
    args = build_run_parser().parse_args(["--config", "experiment.yml"])

    assert args.config == Path("experiment.yml")
    with pytest.raises(SystemExit):
        build_run_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_run_parser().parse_args(["--run-id", "not-accepted"])


def test_load_experiment_config_requires_root_run_id(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yml"
    valid.write_text(
        "run_id: qwen4b-v1\nagent:\n  args: [--pipeline, qwen4b]\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.yml"
    missing.write_text("agent:\n  args: []\n", encoding="utf-8")

    assert load_experiment_config(valid) == {
        "run_id": "qwen4b-v1",
        "agent": {"args": ["--pipeline", "qwen4b"]},
    }
    with pytest.raises(ValueError, match="run_id"):
        load_experiment_config(missing)


def test_evaluate_script_requires_run_id() -> None:
    args = build_evaluate_parser().parse_args(
        [
            "--run-id",
            "existing-run",
            "--runs-dir",
            "/mnt/results/runs",
            "--score-only",
        ]
    )

    assert args.run_id == "existing-run"
    assert args.runs_dir == Path("/mnt/results/runs")
    assert args.score_only is True
    with pytest.raises(SystemExit):
        build_evaluate_parser().parse_args([])


def test_evaluate_script_supports_score_only_external_runs(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, object] = {}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def run(self, *, pipeline_name):
            calls["pipeline_name"] = pipeline_name

    def fake_create(**kwargs):
        calls["create"] = kwargs
        return FakeSession()

    monkeypatch.setattr(
        "scripts.evaluate_experiment.bootstrap_project", lambda root: None
    )
    monkeypatch.setattr(
        "scripts.evaluate_experiment.KedroSession.create", fake_create
    )

    result = evaluate_main(
        [
            "--run-id",
            "existing-run",
            "--runs-dir",
            str(tmp_path),
            "--score-only",
        ]
    )

    assert result == 0
    assert calls["pipeline_name"] == "evaluate"
    assert calls["create"] == {
        "project_path": PROJECT_ROOT,
        "extra_params": {
            "experiment": {
                "run_id": "existing-run",
                "runs_dir": str(tmp_path),
            }
        },
    }


def test_evaluate_script_emits_stable_error_marker(
    monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_experiment.bootstrap_project",
        lambda root: (_ for _ in ()).throw(ValueError("broken config")),
    )

    result = evaluate_main(["--run-id", "existing-run", "--score-only"])

    assert result == 2
    assert capsys.readouterr().err.strip() == (
        "MATHAGENT_EVALUATION_ERROR: ValueError: broken config"
    )


def test_experiment_environments_soft_merge_partial_parameters() -> None:
    loader = OmegaConfigLoader(
        conf_source=str(PROJECT_ROOT / "conf"),
        env="experiments/baseline",
        **CONFIG_LOADER_ARGS,
    )

    parameters = loader["parameters"]

    assert parameters["rag"] == {
        "enabled": False,
        "mode": "similar_conditions",
        "top_k": 3,
        "trigger": "always",
    }
    assert len(parameters["experiment"]["agent"]["benchmarks"]) == 3


def test_runtime_experiment_config_soft_merges_with_base_defaults() -> None:
    loader = OmegaConfigLoader(
        conf_source=str(PROJECT_ROOT / "conf"),
        runtime_params={
            "experiment": {
                "run_id": "qwen4b-v1",
                "agent": {"args": ["--pipeline", "qwen4b"]},
            }
        },
        **CONFIG_LOADER_ARGS,
    )

    experiment = loader["parameters"]["experiment"]

    assert experiment["run_id"] == "qwen4b-v1"
    assert experiment["agent"]["args"] == ["--pipeline", "qwen4b"]
    assert len(experiment["agent"]["benchmarks"]) == 3
    assert experiment["evaluation"]["default"]["script"] == "verify_answers.py"
