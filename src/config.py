"""Configuration loader for Financial LLM project.

Loads config.yaml and provides typed access to all settings.
Central config avoids magic strings scattered across codebase.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"
_config_cache: dict | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration.

    Args:
        config_path: Override path to config file. Defaults to project root config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Override with environment variables where set
    _apply_env_overrides(config)

    if config_path is None:
        _config_cache = config
    return config


def _apply_env_overrides(config: dict) -> None:
    """Override config values with environment variables."""
    env_map = {
        "HF_TOKEN": ("model", "hf_token"),
        "POSTGRES_HOST": ("database", "postgres", "host"),
        "POSTGRES_PORT": ("database", "postgres", "port"),
        "POSTGRES_USER": ("database", "postgres", "user"),
        "POSTGRES_PASSWORD": ("database", "postgres", "password"),
        "POSTGRES_DB": ("database", "postgres", "database"),
        "REDIS_HOST": ("database", "redis", "host"),
        "REDIS_PORT": ("database", "redis", "port"),
    }

    for env_var, key_path in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Navigate to nested key and set
            d = config
            for key in key_path[:-1]:
                d = d.setdefault(key, {})
            # Cast port numbers to int
            if "port" in key_path[-1].lower():
                value = int(value)
            d[key_path[-1]] = value


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
