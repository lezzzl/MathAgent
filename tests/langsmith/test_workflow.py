from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from langsmith.utils import LangSmithNotFoundError

import workflow
from artifact_io import TaskRecord
from bootstrap_stats import SCHEMA_VERSION


def project(
    name: str,
    project_id: str,
    *,
    kind: str = workflow.SOURCE_KIND,
    run_id: str | None = None,
    schema: int = SCHEMA_VERSION,
    day: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        id=project_id,
        start_time=datetime(2025, 1, day, tzinfo=timezone.utc),
        reference_dataset_id="dataset",
        extra={
            "metadata": {
                "kind": kind,
                "schema_version": schema,
                "local_run_id": run_id or name,
            }
        },
    )


class ListingClient:
    def __init__(self, projects):
        self.projects = projects

    def list_projects(self, **_kwargs):
        return iter(self.projects)


def test_compatible_projects_filter_reports_schema_duplicates_and_self() -> None:
    values = [
        project("old", "1", run_id="old"),
        project("duplicate", "2", run_id="old", day=2),
        project("report", "3", kind=workflow.REPORT_KIND),
        project("obsolete", "4", schema=99),
        project("new", "5", run_id="new"),
    ]
    result = workflow.compatible_source_projects(
        ListingClient(values), dataset_id="dataset", exclude_project_id="5"
    )
    assert [item.name for item in result] == ["old"]


def test_report_identity_is_ordered_and_deterministic() -> None:
    left, right = project("left", "1"), project("right", "2")
    name = workflow.report_project_name(left, right, resamples=100, seed=42)
    assert name == workflow.report_project_name(left, right, resamples=100, seed=42)
    assert name != workflow.report_project_name(right, left, resamples=100, seed=42)


def test_native_pairwise_accuracy_scores_wins_ties_and_unresolved() -> None:
    left = SimpleNamespace(id="left", outputs={"correct": False})
    right = SimpleNamespace(id="right", outputs={"correct": True})
    assert workflow._pairwise_accuracy([left, right])["scores"] == {
        "left": 0.0,
        "right": 1.0,
    }
    right.outputs["correct"] = False
    assert workflow._pairwise_accuracy([left, right])["scores"] == {
        "left": 0.5,
        "right": 0.5,
    }
    right.outputs["correct"] = None
    assert workflow._pairwise_accuracy([left, right])["scores"] == {}


def test_publish_native_comparison_uses_langsmith_comparative_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right = project("left", "left-id"), project("right", "right-id")
    comparative = SimpleNamespace(id="comparison-id", name="native-comparison")
    captured = {}

    def evaluate(experiments, **kwargs):
        captured["experiments"] = experiments
        captured.update(kwargs)
        return SimpleNamespace(
            comparative_experiment=comparative,
            url="https://smith/native-comparison",
        )

    monkeypatch.setattr(workflow, "evaluate_comparative", evaluate)
    result, url = workflow.publish_native_comparison(object(), left, right)
    assert result is comparative
    assert url == "https://smith/native-comparison"
    assert captured["experiments"] == ("left-id", "right-id")
    assert captured["evaluators"] == [workflow._pairwise_accuracy]
    assert captured["randomize_order"] is False


def test_native_comparison_is_not_recreated_when_report_tracks_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = project("report", "report-id", kind=workflow.REPORT_KIND)
    report.extra["metadata"]["native_comparative_experiment_id"] = "native-id"
    monkeypatch.setattr(
        workflow,
        "publish_native_comparison",
        lambda *_args: pytest.fail("native comparison must be reused"),
    )
    assert workflow._ensure_native_comparison(
        object(), report, project("left", "left"), project("right", "right")
    ) is report


class ExampleClient:
    def __init__(self, examples=()):
        self.examples = {str(example.id): example for example in examples}
        self.created = []
        self.list_calls = []

    def list_examples(self, *, example_ids=None, **_kwargs):
        self.list_calls.append(example_ids)
        selected = (
            self.examples.values()
            if example_ids is None
            else (
                self.examples[str(example_id)]
                for example_id in example_ids
                if str(example_id) in self.examples
            )
        )
        return iter(selected)

    def create_examples(self, *, examples, **_kwargs):
        self.created.extend(examples)
        for definition in examples:
            self.examples[str(definition["id"])] = SimpleNamespace(
                id=definition["id"],
                inputs=definition.get("inputs"),
                outputs=definition.get("outputs"),
            )


def test_different_model_runs_have_distinct_source_experiments() -> None:
    assert workflow._source_project_name(
        "baseline-qwen35-4b-all-v1"
    ) != workflow._source_project_name("baseline-qwen35-9b-all-v1")


def test_shared_task_has_same_canonical_example_across_model_runs() -> None:
    task = TaskRecord("AIME26", "1", None, "42", "42", None, {})
    assert workflow._example_id(task) == workflow._example_id(task)


def test_existing_canonical_example_is_reused_without_create() -> None:
    example_id = UUID("3aad2e19-d422-595c-8d97-f80ab5c76a9f")
    existing = SimpleNamespace(
        id=example_id,
        inputs={
            "example_key": "AIME26:1",
            "benchmark_name": "AIME26",
            "task_id": "1",
        },
        outputs={"solution": "42"},
    )
    client = ExampleClient([existing])
    synchronized = workflow._get_or_create_examples(
        client,
        "dataset",
        [
            {
                "id": example_id,
                "inputs": existing.inputs,
                "outputs": existing.outputs,
            }
        ],
    )
    assert synchronized == [existing]
    assert client.created == []


def test_only_missing_canonical_examples_are_created() -> None:
    existing_id = UUID("3aad2e19-d422-595c-8d97-f80ab5c76a9f")
    missing_id = UUID("9ca9babb-ca37-5272-8925-55a52e30212d")
    existing = SimpleNamespace(
        id=existing_id, inputs={"report_name": "one"}, outputs={}
    )
    client = ExampleClient([existing])
    synchronized = workflow._get_or_create_examples(
        client,
        "dataset",
        [
            {"id": existing_id, "inputs": {"report_name": "one"}, "outputs": {}},
            {"id": missing_id, "inputs": {"report_name": "two"}, "outputs": {}},
        ],
    )
    assert [example.id for example in synchronized] == [existing_id, missing_id]
    assert [definition["id"] for definition in client.created] == [missing_id]
    assert client.list_calls == [None, None]


class ResolvingClient:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def read_project(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.value


def test_resolve_source_project_uses_name_without_trying_uuid() -> None:
    expected = project("mathagent::run-a", "project-id")
    client = ResolvingClient(expected)
    actual = workflow.resolve_source_project(client, "mathagent::run-a")
    assert actual is expected
    assert client.calls == [{"project_name": "mathagent::run-a"}]


def test_resolve_source_project_uses_uuid_with_id_precedence() -> None:
    identifier = "d853d2f4-b366-45c4-b9ba-1a3864857c37"
    expected = project("source", identifier)
    client = ResolvingClient(expected)
    actual = workflow.resolve_source_project(client, identifier)
    assert actual is expected
    assert client.calls == [{"project_id": UUID(identifier)}]


@pytest.mark.parametrize(
    ("identifier", "expected_kind"),
    [
        ("mathagent::missing", "name"),
        ("d853d2f4-b366-45c4-b9ba-1a3864857c37", "ID"),
    ],
)
def test_resolve_source_project_reports_missing_identifier(
    identifier: str, expected_kind: str
) -> None:
    client = ResolvingClient(error=LangSmithNotFoundError("missing"))
    with pytest.raises(ValueError, match=f"with {expected_kind}"):
        workflow.resolve_source_project(client, identifier)


def test_resolve_source_project_rejects_report_experiment() -> None:
    report = project("report", "report-id", kind=workflow.REPORT_KIND)
    with pytest.raises(ValueError, match="not a compatible MathAgent source"):
        workflow.resolve_source_project(ResolvingClient(report), "report")


def test_publish_compares_every_previous_run_and_collects_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new = project("new", "new", run_id="new")
    previous = [project("a", "a"), project("b", "b"), project("c", "c")]
    monkeypatch.setattr(workflow, "load_local_run", lambda *_args: object())
    monkeypatch.setattr(workflow, "publish_source_run", lambda *_args, **_kwargs: new)
    monkeypatch.setattr(
        workflow, "compatible_source_projects", lambda *_args, **_kwargs: previous
    )

    def compare(_client, left, right, **_kwargs):
        assert right is new
        if left.name == "b":
            raise RuntimeError("broken")
        return project(f"report-{left.name}", f"r-{left.name}"), f"url-{left.name}"

    monkeypatch.setattr(workflow, "publish_comparison", compare)
    _source, reports, failures = workflow.publish_and_compare(
        object(), results_dir=Path("."), run_id="new"
    )
    assert [item[0].name for item in reports] == ["report-a", "report-c"]
    assert [(name, str(error)) for name, error in failures] == [("b", "broken")]


def test_renderer_has_directional_classes_and_safe_dom_api() -> None:
    html = (Path(workflow.__file__).parent / "renderer" / "index.html").read_text()
    assert "strong-positive" in html
    assert "light-negative" in html
    assert "createTextNode" in html
    assert "innerHTML" not in html
