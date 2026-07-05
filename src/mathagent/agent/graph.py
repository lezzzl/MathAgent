from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from mathagent.pipelines.agent_eval.nodes import create_solver_node


@dataclass(frozen=True)
class ModelConfig:
    """Хранит параметры OpenAI-compatible модели."""

    name: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout: float = 600.0
    max_retries: int = 1


class SolverState(TypedDict):
    """Описывает состояние базового математического графа."""

    problem: str
    solution: NotRequired[str]
    usage: NotRequired[dict[str, Any]]
    prompt_version: NotRequired[str]


def create_solver_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    role_name: str = "solver",
) -> Any:
    """Создаёт модель и компилирует граф"""
    model = ChatOpenAI(
        model=model_config.name,
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        temperature=model_config.temperature,
        timeout=model_config.timeout,
        max_retries=model_config.max_retries,
        extra_body={
            "max_tokens": model_config.max_tokens,
            "reasoning_effort": "none",
        },
    )
    graph = StateGraph(SolverState)
    graph.add_node("solve", create_solver_node(model, prompt_path, role_name))
    graph.add_edge(START, "solve")
    graph.add_edge("solve", END)
    return graph.compile()
