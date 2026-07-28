"""Kedro project settings."""

from __future__ import annotations

import os
from typing import Any


def _environment(variable: str, default: Any = None) -> Any:
    """Resolve ``${oc.env:NAME,default}`` without exposing secret values."""

    return os.getenv(variable, default)


CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # Experiment environments contain only the values they change.
    "merge_strategy": {"parameters": "soft"},
    "custom_resolvers": {
        "oc.env": _environment,
    }
}
