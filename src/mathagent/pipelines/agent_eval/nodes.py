from pathlib import Path
from typing import Any, Callable

import yaml


def get_message_usage(message: Any) -> dict[str, Any]:
    """Возвращает token usage и причину завершения вызова модели."""
    usage = dict(getattr(message, "usage_metadata", None) or {})
    response_metadata = getattr(message, "response_metadata", None) or {}
    usage["finish_reason"] = response_metadata.get("finish_reason")
    return usage


def get_message_reasoning(message: Any) -> str | None:
    """Извлекает reasoning из новых и старых форматов ответа vLLM."""
    for attribute in ("reasoning", "reasoning_content"):
        value = getattr(message, attribute, None)
        if isinstance(value, str) and value:
            return value

    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(message, container_name, None) or {}
        for field in ("reasoning", "reasoning_content"):
            value = container.get(field)
            if isinstance(value, str) and value:
                return value
    return None


def create_solver_node(
    model: Any,
    prompt_path: Path,
    role_name: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаёт solver-узел с моделью и выбранной ролью промпта."""
    with prompt_path.open(encoding="utf-8") as stream:
        prompt_config = yaml.safe_load(stream)
    try:
        prompt_version = str(prompt_config["version"])
        role = prompt_config["roles"][role_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid prompt config or role '{role_name}': {prompt_path}"
        ) from exc

    def solve(state: dict[str, Any]) -> dict[str, Any]:
        """Вызывает модель для одной задачи и возвращает обновление state."""
        task_prompt = role["task"].format(problem=state["problem"])
        message = model.invoke(
            [("system", role["system"]), ("human", task_prompt)]
        )
        return {
            "solution": message.content,
            "reasoning": get_message_reasoning(message),
            "usage": get_message_usage(message),
            "prompt_version": prompt_version,
        }

    return solve


def load_prompt_role(prompt_path: Path, role_name: str) -> tuple[str, dict[str, str]]:
    """Загружает нужную роль из YAML-конфига промпта."""
    with prompt_path.open(encoding="utf-8") as stream:
        prompt_config = yaml.safe_load(stream)
    try:
        prompt_version = str(prompt_config["version"])
        role = prompt_config["roles"][role_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid prompt config or role '{role_name}': {prompt_path}"
        ) from exc
    return prompt_version, role


def add_usage(
    state: dict[str, Any],
    role_name: str,
    message: Any,
) -> dict[str, Any]:
    """Добавляет usage конкретного вызова модели в общий словарь usage."""
    usage = dict(state.get("usage", {}))
    usage[role_name] = get_message_usage(message)
    return usage


def add_reasoning(
    state: dict[str, Any],
    role_name: str,
    message: Any,
) -> dict[str, str]:
    """Добавляет reasoning отдельного code-agent узла в общий trace."""
    reasoning = dict(state.get("reasoning", {}))
    message_reasoning = get_message_reasoning(message)
    if message_reasoning is not None:
        reasoning[role_name] = message_reasoning
    return reasoning
