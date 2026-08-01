from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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
from mathagent.pipelines.agent_eval.nodes_react import (
    create_python_tool,
    create_python_tool_node,
    create_react_agent_node,
)
from mathagent.pipelines.agent_eval.nodes_step_code import (
    commit_final_step,
    commit_next_step,
    create_plan_parser_node,
    create_step_coder_node,
    create_step_controller_node,
    create_step_executor_node,
    create_step_finalizer_node,
    create_step_planner_node,
    create_step_repair_node,
    initialize_step_code_state,
    parse_step_output,
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


class StepCodeAgentState(TypedDict):
    """Описывает состояние динамического пошагового code-agent графа."""

    problem: str
    messages: Annotated[list[AnyMessage], add_messages]
    plan_raw: NotRequired[str]
    plan: NotRequired[list[str]]
    current_step: NotRequired[str]
    step_index: NotRequired[int]
    completed_code: NotRequired[list[str]]
    current_code: NotRequired[str]
    stdout: NotRequired[str]
    intermediate_stdout: NotRequired[str]
    stderr: NotRequired[str]
    returncode: NotRequired[int | None]
    timeout: NotRequired[bool]
    marker_found: NotRequired[bool]
    repair_attempt: NotRequired[int]
    coder_format_retry: NotRequired[int]
    output_valid: NotRequired[bool]
    output_error: NotRequired[str | None]
    parsed_output: NotRequired[dict[str, Any]]
    step_result: NotRequired[Any]
    coder_action: NotRequired[str | None]
    coder_next_step: NotRequired[str | None]
    controller_route: NotRequired[str | None]
    finish_reason: NotRequired[str | None]
    attempt_history: NotRequired[list[dict[str, Any]]]
    controller_history: NotRequired[list[dict[str, Any]]]
    planner_trace: NotRequired[dict[str, Any]]
    status: NotRequired[str]
    solution: NotRequired[str | None]
    reasoning: NotRequired[dict[str, str]]
    usage: NotRequired[dict[str, Any]]
    prompt_version: NotRequired[str]
    trace: NotRequired[dict[str, Any]]


class ReactAgentState(TypedDict):
    """Описывает состояние ReAct-цикла с нативным Python tool."""

    problem: str
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: NotRequired[int]
    agent_history: NotRequired[list[dict[str, Any]]]
    tool_history: NotRequired[list[dict[str, Any]]]
    finish_reason: NotRequired[str]
    status: NotRequired[str]
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


def create_step_code_agent_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    max_steps: int = 8,
    max_repairs: int = 2,
    max_coder_format_retries: int = 1,
    execution_timeout: float = 10.0,
    node_generation: NodeGeneration | None = None,
) -> Any:
    """Создаёт динамический граф пошаговой генерации и выполнения кода."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    if max_coder_format_retries < 0:
        raise ValueError("max_coder_format_retries must be non-negative")
    if execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")

    generation = node_generation if node_generation is not None else {}
    graph = StateGraph(StepCodeAgentState)
    graph.add_node("initialize", initialize_step_code_state)
    graph.add_node(
        "planner",
        create_step_planner_node(
            create_node_model(
                model_config,
                prompt_path,
                "planner",
                generation,
            ),
            prompt_path,
            max_steps,
        ),
    )
    graph.add_node("parse_plan", create_plan_parser_node(max_steps))
    graph.add_node(
        "step_coder",
        create_step_coder_node(
            create_node_model(
                model_config,
                prompt_path,
                "step_coder",
                generation,
            ),
            prompt_path,
        ),
    )
    graph.add_node(
        "execute_step",
        create_step_executor_node(execution_timeout),
    )
    graph.add_node("parse_step_output", parse_step_output)
    graph.add_node(
        "controller",
        create_step_controller_node(
            max_steps,
            max_repairs,
            max_coder_format_retries,
        ),
    )
    graph.add_node("commit_next", commit_next_step)
    graph.add_node("commit_finish", commit_final_step)
    graph.add_node(
        "finalizer",
        create_step_finalizer_node(
            create_node_model(
                model_config,
                prompt_path,
                "finalizer",
                generation,
            ),
            prompt_path,
        ),
    )
    if max_repairs > 0:
        graph.add_node(
            "repair",
            create_step_repair_node(
                create_node_model(
                    model_config,
                    prompt_path,
                    "repair",
                    generation,
                ),
                prompt_path,
            ),
        )
        graph.add_edge("repair", "execute_step")

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "planner")
    graph.add_edge("planner", "parse_plan")
    graph.add_edge("parse_plan", "step_coder")
    graph.add_edge("step_coder", "execute_step")
    graph.add_edge("execute_step", "parse_step_output")
    graph.add_edge("parse_step_output", "controller")

    routes = {
        "retry_coder": "step_coder",
        "commit_next": "commit_next",
        "commit_finish": "commit_finish",
        "finalize": "finalizer",
    }
    if max_repairs > 0:
        routes["repair"] = "repair"
    graph.add_conditional_edges(
        "controller",
        lambda state: state["controller_route"],
        routes,
    )
    graph.add_edge("commit_next", "step_coder")
    graph.add_edge("commit_finish", "finalizer")
    graph.add_edge("finalizer", END)

    recursion_limit = (
        8
        + max_steps
        * (6 + 4 * max_repairs + 4 * max_coder_format_retries)
    )
    return graph.compile().with_config({"recursion_limit": recursion_limit})


def create_react_agent_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    max_tool_calls: int = 8,
    execution_timeout: float = 10.0,
    node_generation: NodeGeneration | None = None,
) -> Any:
    """Создаёт ReAct-граф с нативными вызовами изолированного Python tool."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    if execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")

    generation = node_generation if node_generation is not None else {}
    model = create_node_model(
        model_config,
        prompt_path,
        "agent",
        generation,
    )
    python_tool = create_python_tool(execution_timeout)

    graph = StateGraph(ReactAgentState)
    graph.add_node(
        "agent",
        create_react_agent_node(
            model,
            python_tool,
            prompt_path,
            max_tool_calls,
        ),
    )
    graph.add_node("execute_python", create_python_tool_node(python_tool))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        lambda state: (
            "execute_python"
            if state["messages"][-1].tool_calls
            else "finished"
        ),
        {
            "execute_python": "execute_python",
            "finished": END,
        },
    )
    graph.add_edge("execute_python", "agent")
    recursion_limit = 2 * max_tool_calls + 4
    return graph.compile().with_config({"recursion_limit": recursion_limit})
