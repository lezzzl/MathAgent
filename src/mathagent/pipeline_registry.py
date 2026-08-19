"""Реестр пайплайнов проекта.

Kedro автоматически находит все пакеты в mathagent.pipelines, у которых есть
create_pipeline(). Пайплайн по умолчанию (`kedro run` без --pipeline) — benchmarks.
"""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Собирает пайплайны и назначает benchmarks пайплайном по умолчанию."""
    pipelines = find_pipelines()
    pipelines["__default__"] = pipelines.get(
        "benchmarks", sum(pipelines.values(), Pipeline([]))
    )
    return pipelines
