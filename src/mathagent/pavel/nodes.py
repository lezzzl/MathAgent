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


def load_prompt_role(prompt_path: Path, role_name: str) -> tuple[str, dict[str, Any]]:
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


def prompt_has_role(prompt_path: Path, role_name: str) -> bool:
    """Проверяет наличие роли в корректном YAML-конфиге промпта."""
    with prompt_path.open(encoding="utf-8") as stream:
        prompt_config = yaml.safe_load(stream)
    try:
        roles = prompt_config["roles"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid prompt config: {prompt_path}") from exc
    if not isinstance(roles, dict):
        raise ValueError(f"Invalid prompt roles: {prompt_path}")
    return role_name in roles


def load_verification_config(prompt_path: Path) -> dict[str, Any] | None:
    """Загружает ссылку ReAct-промпта на внешний verifier из другой ветки"""
    with prompt_path.open(encoding="utf-8") as stream:
        prompt_config = yaml.safe_load(stream)
    if not isinstance(prompt_config, dict):
        raise ValueError(f"Invalid prompt config: {prompt_path}")

    verification = prompt_config.get("verification")
    if verification is None:
        return None
    if not isinstance(verification, dict):
        raise ValueError(f"Invalid verification config: {prompt_path}")

    prompt_name = verification.get("prompt")
    stepwise_role = verification.get("stepwise_role")
    finalize_role = verification.get("finalize_role")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (prompt_name, stepwise_role, finalize_role)
    ):
        raise ValueError(f"Incomplete verification config: {prompt_path}")

    verifier_prompt_path = prompt_path.parent / prompt_name
    if not verifier_prompt_path.is_file():
        raise ValueError(f"Verifier prompt does not exist: {verifier_prompt_path}")

    with verifier_prompt_path.open(encoding="utf-8") as stream:
        verifier_prompt = yaml.safe_load(stream)
    if not isinstance(verifier_prompt, dict):
        raise ValueError(f"Invalid verifier prompt: {verifier_prompt_path}")
    for role_name in (stepwise_role, finalize_role):
        role = verifier_prompt.get(role_name)
        if not isinstance(role, dict):
            raise ValueError(
                f"Verifier role '{role_name}' is missing: {verifier_prompt_path}"
            )

    return {
        "prompt_path": verifier_prompt_path,
        "stepwise_role": stepwise_role,
        "finalize_role": finalize_role,
    }


def load_prompt_tools(prompt_path: Path) -> dict[str, dict[str, Any]]:
    """Загружает необязательные versioned-описания tools из prompt-конфига."""
    with prompt_path.open(encoding="utf-8") as stream:
        prompt_config = yaml.safe_load(stream)
    if not isinstance(prompt_config, dict):
        raise ValueError(f"Invalid prompt config: {prompt_path}")

    tools = prompt_config.get("tools", {})
    if not isinstance(tools, dict):
        raise ValueError(f"Invalid prompt tools: {prompt_path}")

    normalized: dict[str, dict[str, Any]] = {}
    for tool_name, tool_config in tools.items():
        if not isinstance(tool_name, str) or not isinstance(tool_config, dict):
            raise ValueError(f"Invalid prompt tool config: {prompt_path}")
        description = tool_config.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Prompt tool '{tool_name}' has no description: {prompt_path}"
            )
        normalized[tool_name] = {**tool_config, "description": description.strip()}
    return normalized


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
