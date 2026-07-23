"""Пайплайн agent_eval.

Узлы (create_solver_node и др.) сейчас используются напрямую из
mathagent.agent.graph при сборке LangGraph-графа, поэтому Kedro-пайплайн пока
пустой. Оставлен как точка расширения (например, оффлайн-оценка результатов).
"""

from kedro.pipeline import Pipeline, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """Пока пустой пайплайн — заглушка под будущую оффлайн-оценку."""
    return pipeline([])
