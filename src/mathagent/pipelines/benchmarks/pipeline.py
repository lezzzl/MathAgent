"""Пайплайн benchmarks: один узел-оркестратор, параметризованный из conf/."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import publish_to_langsmith, run_benchmarks


def create_pipeline(**kwargs) -> Pipeline:
    """Гоняет бенчмарки по конфигу и опционально публикует прогон в LangSmith."""
    return pipeline(
        [
            node(
                func=run_benchmarks,
                inputs="params:benchmarks",
                outputs="benchmark_run_summary",
                name="run_benchmarks",
            ),
            # зависит от summary → выполняется ПОСЛЕ прогона; при
            # publish_langsmith=false просто отдаёт {"published": false}
            node(
                func=publish_to_langsmith,
                inputs=["benchmark_run_summary", "params:benchmarks"],
                outputs="langsmith_publish_result",
                name="publish_to_langsmith",
            ),
        ]
    )
