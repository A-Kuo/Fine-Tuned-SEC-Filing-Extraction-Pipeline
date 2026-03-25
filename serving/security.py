"""API key verification for protected routes."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.config import load_config


def assert_api_key_if_configured(request: Request, config: dict | None = None) -> None:
    """Enforce X-API-Key when api_keys is non-empty or require_api_key is true.

    Accepts a pre-loaded config dict to avoid re-parsing config.yaml on every
    request. Falls back to load_config() if not provided.
    """
    cfg = config or load_config()
    keys = cfg.get("serving", {}).get("api_keys") or []
    require = cfg.get("security", {}).get("require_api_key", False)
    if not keys and not require:
        return

    path = request.url.path
    exempt = {"/health", "/docs", "/openapi.json", "/redoc", "/metrics", "/stats", "/webhook/verify"}
    if path in exempt:
        return

    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if keys and key not in keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
