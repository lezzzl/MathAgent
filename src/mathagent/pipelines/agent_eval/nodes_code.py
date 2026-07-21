from pathlib import Path
from typing import Any, Callable

from mathagent.pipelines.agent_eval.nodes import (
    add_reasoning,
    add_usage,
    load_prompt_role,
)
from mathagent.tools.python_executor import PythonExecutor


def create_planner_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт план решения задачи через Python и SymPy"""
    prompt_version, role = load_prompt_role(prompt_path, "planner")

    def plan(state: dict[str, Any]) -> dict[str, Any]:
        """Вызывает planner и добавляет его результат в state"""
        task_prompt = role["task"].format(problem=state["problem"])
        message = model.invoke(
            [("system", role["system"]), ("human", task_prompt)]
        )
        return {
            "plan": message.content,
            "reasoning": add_reasoning(state, "planner", message),
            "usage": add_usage(state, "planner", message),
            "prompt_version": prompt_version,
        }

    return plan


def create_coder_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт Python и SymPy код по задаче и плану"""
    prompt_version, role = load_prompt_role(prompt_path, "coder")

    def code(state: dict[str, Any]) -> dict[str, Any]:
        """Вызывает coder и сохраняет сгенерированный код"""
        task_prompt = role["task"].format(
            problem=state["problem"],
            plan=state["plan"],
        )
        message = model.invoke(
            [("system", role["system"]), ("human", task_prompt)]
        )
        return {
            "code": message.content,
            "reasoning": add_reasoning(state, "coder", message),
            "usage": add_usage(state, "coder", message),
            "prompt_version": prompt_version,
        }

    return code


def create_code_executor_node(
    timeout: float = 10.0,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт ноду запуска Python-кода с ограничением времени"""
    executor = PythonExecutor(timeout=timeout)

    def execute(state: dict[str, Any]) -> dict[str, Any]:
        """Выполняет последний код и добавляет попытку в execution_history"""
        execution = executor.run(state["code"]).to_dict()
        execution_history = list(state.get("execution_history", []))
        execution_history.append(
            {
                "attempt": state.get("repair_attempt", 0),
                "code": state["code"],
                **execution,
            }
        )
        return {
            **execution,
            "execution_history": execution_history,
        }

    return execute


def create_repair_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт исправленный код после неуспешного выполнения"""
    prompt_version, role = load_prompt_role(prompt_path, "repair")

    def repair(state: dict[str, Any]) -> dict[str, Any]:
        """Передаёт repair только последнюю попытку и увеличивает её номер"""
        attempt = state.get("repair_attempt", 0) + 1
        task_prompt = role["task"].format(
            problem=state["problem"],
            plan=state["plan"],
            code=state["code"],
            stdout=state.get("stdout", ""),
            stderr=state.get("stderr", ""),
        )
        message = model.invoke(
            [("system", role["system"]), ("human", task_prompt)]
        )
        return {
            "code": message.content,
            "repair_attempt": attempt,
            "reasoning": add_reasoning(state, f"repair_{attempt}", message),
            "usage": add_usage(state, f"repair_{attempt}", message),
            "prompt_version": prompt_version,
        }

    return repair


def create_finalizer_node(
    model: Any,
    prompt_path: Path,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт финальный ответ и общий trace code-agent графа"""
    prompt_version, role = load_prompt_role(prompt_path, "finalizer")

    def finalize(state: dict[str, Any]) -> dict[str, Any]:
        """Извлекает ответ из последнего stdout и stderr"""
        task_prompt = role["task"].format(
            stdout=state.get("stdout", ""),
            stderr=state.get("stderr", ""),
        )
        message = model.invoke(
            [("system", role["system"]), ("human", task_prompt)]
        )
        return {
            "solution": message.content,
            "reasoning": add_reasoning(state, "finalizer", message),
            "usage": add_usage(state, "finalizer", message),
            "prompt_version": prompt_version,
            "trace": {
                "plan": state.get("plan", ""),
                "execution_history": state.get("execution_history", []),
            },
        }

    return finalize
