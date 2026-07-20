from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from mathagent.agent.vllm_chat import ChatVLLM
from mathagent.pipelines.agent_eval.nodes import (
    create_solver_node,
)


@dataclass(frozen=True)
class ModelConfig:
    """Хранит параметры OpenAI-compatible модели."""

    name: str
    base_url: str
    api_key: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    seed: int = 42
    thinking: bool = True
    max_tokens: int = 65536
    timeout: float = 7200.0
    max_retries: int = 1


class SolverState(TypedDict):
    """Описывает состояние базового математического графа."""

    problem: str
    solution: NotRequired[str]
    reasoning: NotRequired[str | None]
    usage: NotRequired[dict[str, Any]]
    prompt_version: NotRequired[str]

def create_solver_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    role_name: str = "solver",
) -> Any:
    """Создаёт модель и компилирует граф"""
    model = ChatVLLM(
        model=model_config.name,
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        presence_penalty=model_config.presence_penalty,
        seed=model_config.seed,
        timeout=model_config.timeout,
        max_retries=model_config.max_retries,
        extra_body={
            "max_tokens": model_config.max_tokens,
            "top_k": model_config.top_k,
            "min_p": model_config.min_p,
            "repetition_penalty": model_config.repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": model_config.thinking,
            },
        },
    )
    graph = StateGraph(SolverState)
    graph.add_node("solve", create_solver_node(model, prompt_path, role_name))
    graph.add_edge(START, "solve")
    graph.add_edge("solve", END)
    return graph.compile()
