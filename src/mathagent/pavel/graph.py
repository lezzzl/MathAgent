from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mathagent.agent.vllm_chat import ChatVLLM
from mathagent.pavel.nodes import (
    create_solver_node,
    load_prompt_tools,
    load_prompt_role,
    load_verification_config,
    prompt_has_role,
)
from mathagent.pavel.nodes_code import (
    create_code_executor_node,
    create_coder_node,
    create_finalizer_node,
    create_planner_node,
    create_repair_node,
)
from mathagent.pavel.nodes_react import (
    create_react_code_repair_node,
    create_react_agent_node,
    create_react_planner_node,
    create_react_precheck_node,
    create_react_repair_executor_node,
    record_react_tool_call,
    route_after_react_agent,
    route_after_react_execution,
    route_after_react_precheck,
    route_after_react_repair,
)
from mathagent.pavel.nodes_verifier import (
    create_react_verifier_node,
    route_after_react_verification,
)
from mathagent.tools.final_answer import (
    create_final_answer_tool,
    create_final_solution_tool,
)
from mathagent.tools.cot import create_cot_tool
from mathagent.tools.python_tools import (
    create_notebook_python_tool,
    create_notebook_sympy_tool,
    create_python_tool,
    create_repair_tool,
)
from mathagent.tools.python_executor import NotebookSessionManager
from mathagent.tools.tool_review import create_tool_review_tool


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


NodeGeneration = dict[str, dict[str, bool | int | float]]


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


class ReactAgentState(TypedDict):
    """Описывает состояние ReAct-цикла с нативным Python tool."""

    problem: str
    messages: Annotated[list[AnyMessage], add_messages]
    python_session_id: NotRequired[str]
    plan: NotRequired[str]
    planner_trace: NotRequired[dict[str, Any]]
    tool_call_count: NotRequired[int]
    agent_history: NotRequired[list[dict[str, Any]]]
    tool_history: NotRequired[list[dict[str, Any]]]
    repair_history: NotRequired[list[dict[str, Any]]]
    repair_attempt: NotRequired[int]
    repair_code: NotRequired[str]
    repair_execution: NotRequired[dict[str, Any]]
    repair_tool_call_id: NotRequired[str]
    repair_tool_name: NotRequired[str]
    repair_context: NotRequired[str]
    repair_feedback: NotRequired[str]
    repair_candidate_validation: NotRequired[dict[str, Any]]
    repair_model_usage: NotRequired[dict[str, Any]]
    repair_model_latency: NotRequired[float]
    precheck_history: NotRequired[list[dict[str, Any]]]
    precheck_rejection_count: NotRequired[int]
    force_final_reason: NotRequired[str]
    proposed_answer: NotRequired[str]
    proposed_solution: NotRequired[str]
    proposed_tool_call_id: NotRequired[str]
    solution_attempts: NotRequired[list[dict[str, Any]]]
    verification_round: NotRequired[int]
    format_retry_count: NotRequired[int]
    had_format_recovery: NotRequired[bool]
    finish_reason: NotRequired[str]
    status: NotRequired[str]
    solution: NotRequired[str]
    reasoning: NotRequired[dict[str, str]]
    usage: NotRequired[dict[str, Any]]
    prompt_version: NotRequired[str]
    trace: NotRequired[dict[str, Any]]


class ManagedNotebookGraph:
    """Добавляет отдельные notebook-сессии каждому graph invocation."""

    def __init__(
        self,
        graph: Any,
        managers: NotebookSessionManager | list[NotebookSessionManager],
    ) -> None:
        self.graph = graph
        self.managers = (
            [managers] if isinstance(managers, NotebookSessionManager) else managers
        )
        if not self.managers:
            raise ValueError("ManagedNotebookGraph requires a session manager")
        self.manager = self.managers[0]

    def invoke(
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Создаёт сессию задачи и гарантированно закрывает её после графа."""
        session_id = uuid4().hex
        try:
            return self.graph.invoke(
                {**input_state, "python_session_id": session_id},
                config=config,
                **kwargs,
            )
        finally:
            for manager in self.managers:
                manager.close(session_id)

    def __getattr__(self, name: str) -> Any:
        """Делегирует диагностические методы исходному compiled graph."""
        return getattr(self.graph, name)


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


def create_react_agent_graph(
    model_config: ModelConfig,
    prompt_path: Path,
    max_tool_calls: int = 8,
    max_tool_repairs: int = 5,
    max_precheck_rejections: int = 3,
    max_verification_rounds: int = 2,
    execution_timeout: float = 10.0,
    node_generation: NodeGeneration | None = None,
) -> Any:
    """Создаёт ReAct-граф с набором tools из выбранной версии промпта."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    if max_tool_repairs < 0:
        raise ValueError("max_tool_repairs must be non-negative")
    if max_precheck_rejections < 1:
        raise ValueError("max_precheck_rejections must be positive")
    if max_verification_rounds < 1:
        raise ValueError("max_verification_rounds must be positive")
    if execution_timeout <= 0:
        raise ValueError("execution_timeout must be positive")

    generation = node_generation if node_generation is not None else {}
    agent_model = create_node_model(
        model_config,
        prompt_path,
        "agent",
        generation,
    )
    cot_enabled = prompt_has_role(prompt_path, "cot")
    precheck_enabled = prompt_has_role(prompt_path, "tool_checker")
    planner_enabled = prompt_has_role(prompt_path, "planner")
    verification_config = load_verification_config(prompt_path)
    verification_enabled = verification_config is not None
    structured_repair_configured = cot_enabled and prompt_has_role(
        prompt_path,
        "repair",
    )
    structured_repair_enabled = (
        structured_repair_configured and max_tool_repairs > 0
    )
    prompt_tools = load_prompt_tools(prompt_path)
    unsupported_tools = set(prompt_tools) - {"python", "sympy"}
    if unsupported_tools:
        names = ", ".join(sorted(unsupported_tools))
        raise ValueError(f"Unsupported ReAct execution tools: {names}")
    planner_model = None
    if planner_enabled:
        planner_model = create_node_model(
            model_config,
            prompt_path,
            "planner",
            generation,
        )
    structured_repair_model = None
    if structured_repair_enabled:
        structured_repair_model = create_node_model(
            model_config,
            prompt_path,
            "repair",
            generation,
        )
    cot_tool = None
    repair_tool = None
    notebook_managers: list[NotebookSessionManager] = []
    execution_tools: list[Any]
    if cot_enabled:
        cot_model = create_node_model(
            model_config,
            prompt_path,
            "cot",
            generation,
        )
        cot_tool = create_cot_tool(cot_model, prompt_path)
        notebook_manager = NotebookSessionManager(timeout=execution_timeout)
        notebook_managers.append(notebook_manager)
        python_description = prompt_tools.get("python", {}).get("description")
        python_tool = create_notebook_python_tool(
            notebook_manager,
            **(
                {"description": python_description}
                if isinstance(python_description, str)
                else {}
            ),
        )
        execution_tools = [python_tool]
        if "sympy" in prompt_tools:
            execution_tools.append(
                create_notebook_sympy_tool(
                    notebook_manager,
                    prompt_tools["sympy"]["description"],
                )
            )
        executable_tools = [cot_tool, *execution_tools]
    else:
        python_tool = create_python_tool(execution_timeout)
        execution_tools = [python_tool]
        repair_tool = create_repair_tool(execution_timeout)
        executable_tools = [python_tool, repair_tool]
    terminal_tool = (
        create_final_solution_tool()
        if verification_enabled
        else create_final_answer_tool()
    )
    verifier_stepwise_model = None
    verifier_finalize_model = None
    if verification_enabled:
        if verification_config is None:
            raise ValueError("Verification config is missing")
        stepwise_node_name = str(verification_config["stepwise_role"])
        finalize_node_name = str(verification_config["finalize_role"])
        generation[stepwise_node_name] = {
            "thinking": False,
            "max_tokens": 20000,
            "temperature": 0.1,
        }
        generation[finalize_node_name] = {
            "thinking": False,
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        verifier_stepwise_model = create_model(
            replace(
                model_config,
                thinking=False,
                temperature=0.1,
                max_tokens=20000,
            )
        )
        verifier_finalize_model = create_model(
            replace(
                model_config,
                thinking=False,
                temperature=0.1,
                max_tokens=4096,
            )
        )
    review_tool = None
    precheck_model = None
    if precheck_enabled:
        precheck_model = create_node_model(
            model_config,
            prompt_path,
            "tool_checker",
            generation,
        )
        review_tool = create_tool_review_tool()

    graph = StateGraph(ReactAgentState)
    graph.add_node(
        "agent",
        create_react_agent_node(
            agent_model,
            execution_tools,
            terminal_tool,
            prompt_path,
            max_tool_calls,
            repair_tool=repair_tool,
            cot_tool=cot_tool,
            precheck_enabled=precheck_enabled,
            planner_enabled=planner_enabled,
            structured_repair_enabled=structured_repair_enabled,
            verification_enabled=verification_enabled,
        ),
    )
    if planner_model is not None:
        graph.add_node(
            "planner",
            create_react_planner_node(planner_model, prompt_path),
        )
    if precheck_model is not None and review_tool is not None:
        graph.add_node(
            "precheck",
            create_react_precheck_node(
                precheck_model,
                review_tool,
                prompt_path,
                max_precheck_rejections,
            ),
        )
    graph.add_node(
        "tools",
        ToolNode(executable_tools, handle_tool_errors=False),
    )
    graph.add_node(
        "record_tool",
        partial(
            record_react_tool_call,
            structured_repair_enabled=structured_repair_enabled,
            max_tool_repairs=max_tool_repairs,
        ),
    )
    if structured_repair_model is not None:
        if not notebook_managers:
            raise ValueError("Structured repair requires a notebook manager")
        graph.add_node(
            "code_repair",
            create_react_code_repair_node(
                structured_repair_model,
                prompt_path,
            ),
        )
        graph.add_node(
            "execute_repair",
            create_react_repair_executor_node(
                notebook_managers[0],
                max_tool_repairs,
            ),
        )
    if verifier_stepwise_model is not None and verifier_finalize_model is not None:
        if verification_config is None:
            raise ValueError("Verification config is missing")
        graph.add_node(
            "verifier",
            create_react_verifier_node(
                verifier_stepwise_model,
                verifier_finalize_model,
                prompt_path,
                Path(verification_config["prompt_path"]),
                str(verification_config["stepwise_role"]),
                str(verification_config["finalize_role"]),
                max_verification_rounds,
            ),
        )
    if planner_enabled:
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "agent")
    else:
        graph.add_edge(START, "agent")
    agent_routes = {
        "tools": "precheck" if precheck_enabled else "tools",
        "agent": "agent",
        "finished": END,
    }
    if verification_enabled:
        agent_routes["verify"] = "verifier"
    graph.add_conditional_edges(
        "agent",
        route_after_react_agent,
        agent_routes,
    )
    if verification_enabled:
        graph.add_conditional_edges(
            "verifier",
            route_after_react_verification,
            {
                "agent": "agent",
                "finished": END,
            },
        )
    if precheck_enabled:
        graph.add_conditional_edges(
            "precheck",
            route_after_react_precheck,
            {
                "tools": "tools",
                "agent": "agent",
            },
        )
    graph.add_edge("tools", "record_tool")
    if structured_repair_enabled and max_tool_repairs > 0:
        graph.add_conditional_edges(
            "record_tool",
            route_after_react_execution,
            {
                "repair": "code_repair",
                "agent": "agent",
            },
        )
        graph.add_edge("code_repair", "execute_repair")
        graph.add_conditional_edges(
            "execute_repair",
            route_after_react_repair,
            {
                "repair": "code_repair",
                "agent": "agent",
            },
        )
    else:
        graph.add_edge("record_tool", "agent")
    recursion_limit = (
        (4 if precheck_enabled else 3) * max_tool_calls
        + 2 * max_tool_calls * max_tool_repairs
        + 2 * max_precheck_rejections
        + (1 if planner_enabled else 0)
        + 2 * max_verification_rounds
        + 6
    )
    compiled_graph = graph.compile().with_config({"recursion_limit": recursion_limit})
    if notebook_managers:
        return ManagedNotebookGraph(compiled_graph, notebook_managers)
    return compiled_graph
