"""Project pipeline registration."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from mathagent.pipelines.experiment.pipeline import (
    create_agent_pipeline,
    create_compare_pipeline,
    create_evaluate_compare_pipeline,
    create_evaluate_pipeline,
    create_experiment_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Expose complete and independently runnable experiment stages."""

    experiment = create_experiment_pipeline()
    return {
        "__default__": experiment,
        "experiment": experiment,
        "agent": create_agent_pipeline(),
        "evaluate": create_evaluate_pipeline(),
        "compare": create_compare_pipeline(),
        "evaluate_compare": create_evaluate_compare_pipeline(),
    }
