"""Structured logging setup (JSON or text) with optional request ID context."""

from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from typing import Any

from loguru import logger

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def configure_logging(level: str = "INFO", fmt: str = "text", include_request_id: bool = True) -> None:
    """Remove default loguru handler and add configured sink."""
    logger.remove()
    level = level.upper()

    if fmt == "json":

        def json_sink(message: Any) -> None:
            record = message.record
            payload = {
                "time": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["module"],
                "function": record["function"],
            }
            if include_request_id:
                rid = request_id_var.get()
                if rid:
                    payload["request_id"] = rid
            print(json.dumps(payload, default=str), file=sys.stderr)

        logger.add(json_sink, level=level)
    else:
        fmt_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        logger.add(sys.stderr, format=fmt_str, level=level)


def set_request_id(rid: str | None) -> None:
    request_id_var.set(rid)
