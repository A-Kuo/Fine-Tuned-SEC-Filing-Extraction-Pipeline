"""A/B routing: assign requests to primary or challenger model paths."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.config import load_config


@dataclass
class ABAssignment:
    """Which model path to use for this request."""

    use_challenger: bool
    model_path: str
    model_version_label: str


def assign_for_request(
    filing_id: str | None,
    header_override: str | None,
    config: dict | None = None,
) -> ABAssignment:
    """Decide primary vs challenger using config and optional X-Model-Version header."""
    config = config or load_config()
    ab = config.get("ab_test", {})
    if not ab.get("enabled", False):
        primary = ab.get("primary_model_path") or config["model"]["adapter_path"]
        return ABAssignment(False, primary, "primary")

    primary = ab.get("primary_model_path") or config["model"]["adapter_path"]
    challenger = ab.get("challenger_model_path", primary)
    split = float(ab.get("traffic_split", 0.1))

    if header_override == "challenger":
        return ABAssignment(True, challenger, "challenger")
    if header_override == "primary":
        return ABAssignment(False, primary, "primary")

    # Deterministic hash split by filing_id
    key = filing_id or "default"
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    use_c = (h % 10000) / 10000.0 < split
    if use_c:
        return ABAssignment(True, challenger, "challenger")
    return ABAssignment(False, primary, "primary")
