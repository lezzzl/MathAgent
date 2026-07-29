from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from mathagent.agent.vllm_chat import ChatVLLM
from mathagent.pipelines.agent_eval.nodes import (
    create_solver_node,
    load_prompt_role,
)
from mathagent.pipelines.agent_eval.nodes_code import (
    create_code_executor_node,
    create_coder_node,
    create_finalizer_node,
    create_planner_node,
    create_repair_node,
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


NodeGeneration = dict[str, dict[str, bool | int]]


class SolverState(TypedDict):
    """Описывает состояние базового математического графа."""

    problem: str
    solution: NotRequired[str]
    reasoning: NotRequired[str | None]
    usage: NotRequired[dict[str, Any]]
    prompt_version: NotRequired[str]


class CodeAgentState(TypedDict):
    """Описывает состояние SymCode-like графа"""

    problem: str
    plan: NotRequired[str]
    code: NotRequired[str]
    stdout: NotRequired[str]
    stderr: NotRequired[str]
    returncode: NotRequired[int | None]
    timeout: NotRequired[bool]
    execution_history: NotRequired[list[dict[str, Any]]]
    repair_attempt: NotRequired[int]
    solution: NotRequired[str]
    reasoning: NotRequired[dict[str, str]]
    usage: NotRequired[dict[str, Any]]
    prompt_version: NotRequired[str]
    trace: NotRequired[dict[str, Any]]


def create_model(model_config: ModelConfig) -> ChatVLLM:
    """Создаёт общую модель для solver и code-agent графов"""
    return ChatVLLM(
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


def create_node_model(
    model_config: ModelConfig,
    prompt_path: Path,
    node_name: str,
    node_generation: NodeGeneration,
    role_name: str | None = None,
) -> ChatVLLM:
    """Создаёт модель с настройками роли и регистрирует их для manifest"""
    role_name = role_name or node_name
    _, role = load_prompt_role(prompt_path, role_name)
    thinking = role.get("thinking", model_config.thinking)
    max_tokens = role.get("max_tokens", model_config.max_tokens)

    if not isinstance(thinking, bool):
        raise ValueError(
            f"Role '{role_name}' has invalid thinking value in {prompt_path}: "
            "expected boolean"
        )
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError(
            f"Role '{role_name}' has invalid max_tokens value in {prompt_path}: "
            "expected integer"
        )
    if max_tokens <= 0:
        raise ValueError(
            f"Role '{role_name}' has invalid max_tokens value in {prompt_path}: "
            "expected a positive integer"
        )

    node_generation[node_name] = {
        "thinking": thinking,
        "max_tokens": max_tokens,
    }
    return create_model(
        replace(
            model_config,
            thinking=thinking,
            max_tokens=max_tokens,
        )
    )


def create_solver_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    role_name: str = "solver",
    node_generation: NodeGeneration | None = None,
) -> Any:
    """Создаёт модель и компилирует граф"""
    generation = node_generation if node_generation is not None else {}
    model = create_node_model(
        model_config,
        prompt_path,
        role_name,
        generation,
    )
    graph = StateGraph(SolverState)
    graph.add_node("solve", create_solver_node(model, prompt_path, role_name))
    graph.add_edge(START, "solve")
    graph.add_edge("solve", END)
    return graph.compile()


def create_code_agent_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    max_repairs: int = 5,
    execution_timeout: float = 10.0,
    node_generation: NodeGeneration | None = None,
) -> Any:
    """Создаёт code-agent граф с ограниченным циклом исправления кода"""
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    if execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")

    def route_after_execution(state: CodeAgentState) -> str:
        """Выбирает repair после ошибки или переход к finalizer"""
        execution_failed = (
            state.get("returncode") != 0 or state.get("timeout", False)
        )
        attempts_left = state.get("repair_attempt", 0) < max_repairs
        return "repair" if execution_failed and attempts_left else "finalize"

    generation = node_generation if node_generation is not None else {}
    graph = StateGraph(CodeAgentState)
    graph.add_node(
        "plan",
        create_planner_node(
            create_node_model(
                model_config,
                prompt_path,
                "planner",
                generation,
            ),
            prompt_path,
        ),
    )
    graph.add_node(
        "code",
        create_coder_node(
            create_node_model(
                model_config,
                prompt_path,
                "coder",
                generation,
            ),
            prompt_path,
        ),
    )
    graph.add_node(
        "execute",
        create_code_executor_node(timeout=execution_timeout),
    )
    graph.add_node(
        "finalize",
        create_finalizer_node(
            create_node_model(
                model_config,
                prompt_path,
                "finalizer",
                generation,
            ),
            prompt_path,
        ),
    )
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "code")
    graph.add_edge("code", "execute")

    if max_repairs > 0:
        graph.add_node(
            "repair",
            create_repair_node(
                create_node_model(
                    model_config,
                    prompt_path,
                    "repair",
                    generation,
                ),
                prompt_path,
            ),
        )
        graph.add_conditional_edges(
            "execute",
            route_after_execution,
            {
                "repair": "repair",
                "finalize": "finalize",
            },
        )
        graph.add_edge("repair", "execute")
    else:
        graph.add_edge("execute", "finalize")

    graph.add_edge("finalize", END)
    return graph.compile()
