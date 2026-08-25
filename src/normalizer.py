from __future__ import annotations

import re
from src.schemas import MetricRecord


MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}


def parse_numeric_value(raw: str | int | float | None) -> float | int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return raw

    text = raw.lower().replace("$", "").replace(",", "").strip()
    multiplier = 1

    for token, value in MULTIPLIERS.items():
        if token in text:
            multiplier = value
            text = text.replace(token, "").strip()
            break

    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        return None

    number = float(match.group(0)) * multiplier
    return int(number) if number.is_integer() else number


def normalize_metric(
    name: str,
    raw_value: str | int | float | None,
    *,
    unit: str | None = "usd",
    period: str | None = None,
    segment: str | None = None,
    method: str = "llm",
    confidence: float = 0.5,
    source_section: str | None = None,
    evidence_text: str | None = None,
) -> MetricRecord:
    return MetricRecord(
        name=name,
        value=parse_numeric_value(raw_value),
        unit=unit,
        period=period,
        segment=segment,
        method=method,
        confidence=confidence,
        source_section=source_section,
        evidence_text=evidence_text,
    )