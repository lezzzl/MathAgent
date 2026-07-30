from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Role:
    name: str
    system_prompt: str
    user_template: str
    prefill: str = ""
    stop: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path, role_key: str) -> "Role":
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if role_key not in config:
            raise KeyError(
                f"Роль '{role_key}' не найдена в '{path}'. "
                f"Доступные роли: {list(config.keys())}"
            )

        node = config[role_key]
        return cls(
            name=role_key,
            system_prompt=node["system_prompt"],
            user_template=node["user_prompt_template"],
            prefill=node.get("prefill", "") or "",
            stop=node.get("stop", []) or [],
        )

    def build_messages(self, **kwargs: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(**kwargs)},
        ]
        if self.prefill:
            messages.append({"role": "assistant", "content": self.prefill})
        return messages