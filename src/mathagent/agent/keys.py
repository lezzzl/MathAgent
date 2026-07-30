"""Резолвинг секретов/идентификаторов Yandex AI Studio без хранения в репозитории.

В конфиге (parameters.yml -> agent) указывается НЕ ключ, а откуда его взять:
  api_key_env:   имя переменной окружения с ключом  (приоритетнее)
  api_key_file:  путь к файлу, где лежит ключ
  folder_id_env: имя переменной окружения с folder_id каталога
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_api_key(agent_cfg: dict) -> str:
    """Вернуть API-ключ Yandex AI Studio из env-переменной или файла."""
    env_name = agent_cfg.get("api_key_env")
    if env_name:
        key = os.environ.get(env_name)
        if key:
            return key.strip()

    key_file = agent_cfg.get("api_key_file")
    if key_file:
        path = Path(key_file).expanduser()
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "API-ключ не найден. Задай переменную окружения "
        f"'{env_name or 'YC_API_KEY'}' (например в .env) "
        "или укажи agent.api_key_file в parameters.yml."
    )


def resolve_folder_id(agent_cfg: dict) -> str:
    """Вернуть folder_id каталога Yandex Cloud из env-переменной."""
    env_name = agent_cfg.get("folder_id_env", "YC_FOLDER_ID")
    folder_id = os.environ.get(env_name)
    if not folder_id:
        raise RuntimeError(
            f"folder_id не найден. Задай переменную окружения '{env_name}' "
            "(например в .env)."
        )
    return folder_id.strip()


def build_model_uri(agent_cfg: dict) -> str:
    """Собрать model_uri вида gpt://<folder_id>/<model> для Yandex AI Studio."""
    return f"gpt://{resolve_folder_id(agent_cfg)}/{agent_cfg['model']}"
