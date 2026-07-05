from pathlib import Path
from typing import Any, Callable

import yaml


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
            "usage": getattr(message, "usage_metadata", None) or {},
            "prompt_version": prompt_version,
        }

    return solve
