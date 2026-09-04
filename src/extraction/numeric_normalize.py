"""Shared numeric-string normalization: '$383.3 billion' -> 383300000000.

Before this module, this logic existed in two places that could silently
drift apart: src/extraction/normalizer.py::parse_numeric_value() (used by
the extraction pipeline) and src/storage/database.py::PostgresStorage.
_parse_financial() (used by the flat-schema storage layer, architecture A).
Both are now thin wrappers around this one implementation -- see each for
their re-export/delegation.
"""

from __future__ import annotations

import re

MULTIPLIERS = {
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}


def parse_numeric_value(raw: str | int | float | None) -> float | int | None:
    """Parse a financial figure string into a plain number.

    Handles '$5.23', '$12.1 million', '$383.3 billion', '$1,234,567',
    negative numbers, and already-numeric input (passthrough). Returns None
    if no number can be found.
    """
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


def fuzzy_numeric_match(predicted: str | int | float | None, truth: str | int | float | None, tolerance: float = 0.05) -> bool:
    """True if two numeric-ish values (possibly formatted differently, e.g.
    '$383.3 billion' vs '383300000000') agree within `tolerance` relative
    error. Both None is a match (both correctly abstained); exactly one None
    is not."""
    pred_num = parse_numeric_value(predicted)
    truth_num = parse_numeric_value(truth)

    if pred_num is None and truth_num is None:
        return True
    if pred_num is None or truth_num is None:
        return False
    if truth_num == 0:
        return pred_num == 0

    return abs(pred_num - truth_num) / abs(truth_num) <= tolerance
