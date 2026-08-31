"""Project configuration loader.

Loads config.yaml from the repository root and applies selected environment
variable overrides. Both snake_case and legacy compact aliases are provided
during the repository migration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_config_cache: dict[str, Any] | None = None


def get_project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply supported environment-variable overrides in place."""
    env_map: dict[str, tuple[str, ...]] = {
        "HF_TOKEN": ("model", "hf_token"),
        "HFTOKEN": ("model", "hf_token"),
        "POSTGRES_HOST": ("database", "postgres", "host"),
        "POSTGRES_PORT": ("database", "postgres", "port"),
        "POSTGRES_USER": ("database", "postgres", "user"),
        "POSTGRES_PASSWORD": ("database", "postgres", "password"),
        "POSTGRES_DB": ("database", "postgres", "database"),
        "REDIS_HOST": ("database", "redis", "host"),
        "REDIS_PORT": ("database", "redis", "port"),
        "EDGAR_USER_AGENT": ("edgar", "user_agent"),
        "EDGAR_RPS": ("edgar", "requests_per_second"),
        "API_KEYS": ("serving", "api_keys"),
        "WEBHOOK_SIGNING_SECRET": ("security", "webhook_signing_secret"),
        "LOG_FORMAT": ("logging", "format"),
        "LOG_LEVEL": ("logging", "level"),
        "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
        "KAGGLE_USERNAME": ("kaggle", "username"),
        "KAGGLE_KEY": ("kaggle", "key"),
    }

    integer_paths = {
        ("database", "postgres", "port"),
        ("database", "redis", "port"),
        ("edgar", "requests_per_second"),
    }

    for env_name, key_path in env_map.items():
        value = os.getenv(env_name)
        if value is None:
            continue

        target = config
        for key in key_path[:-1]:
            target = target.setdefault(key, {})

        if key_path in integer_paths:
            target[key_path[-1]] = int(value)
        elif key_path == ("serving", "api_keys"):
            target[key_path[-1]] = [
                item.strip() for item in value.split(",") if item.strip()
            ]
        else:
            target[key_path[-1]] = value


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml and apply environment-variable overrides."""
    global _config_cache

    if config_path is None and _config_cache is not None:
        return _config_cache

    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    _apply_env_overrides(config)

    if config_path is None:
        _config_cache = config

    return config


# Compatibility aliases for existing imports during the naming migration.
loadconfig = load_config
getprojectroot = get_project_root