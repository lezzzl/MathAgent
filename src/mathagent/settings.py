"""Настройки Kedro-проекта.

Оставляем дефолты Kedro 0.19: конфиги грузит OmegaConfigLoader, поэтому
работают резолверы ${globals:...} (при наличии globals.yml) и ${oc.env:VAR,def}.
Реестр пайплайнов — mathagent.pipeline_registry.register_pipelines.
"""

# Явно фиксируем OmegaConfigLoader (в 0.19 он и так по умолчанию) — на случай
# смены версии Kedro, чтобы резолверы в conf/ не отвалились.
import os  # noqa: E402

from kedro.config import OmegaConfigLoader  # noqa: E402


def _oc_env(key: str, default: str | None = None) -> str | None:
    """Резолвер ${oc.env:VAR,default}. Kedro по умолчанию его вырезает (ради
    безопасности), а globals.yml/parameters.yml им пользуются для переопределения
    имени модели, base_url и run_name из окружения — поэтому включаем явно."""
    return os.environ.get(key, default)


CONFIG_LOADER_CLASS = OmegaConfigLoader
# base_env/default_run_env ОБЯЗАТЕЛЬНЫ: без них base_env="" и лоадер сканирует
# весь conf/ (включая conf/experiments/*), падая на дублях ключей.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "custom_resolvers": {"oc.env": _oc_env},
}
