"""Error taxonomy for classifying why a predicted field didn't match ground
truth -- turns "wrong" into a specific, actionable reason code instead of a
single undifferentiated failure count.
"""

from __future__ import annotations

from typing import Literal

from src.extraction.numeric_normalize import fuzzy_numeric_match, parse_numeric_value

ErrorTag = Literal[
    "formatting_normalization_failure",
    "numeric_scaling_failure",
    "wrong_field",
    "missing_required_field",
    "hallucinated_field",
    "parser_recovery_failure",
]

NUMERIC_FIELDS = {"revenue", "net_income", "total_assets", "total_liabilities", "eps"}

# Common unit-confusion factors: thousand/million/billion mixups.
_SCALING_FACTORS = (1_000, 1_000_000, 1_000_000_000)
_SCALING_TOLERANCE = 0.02  # 2% -- catches "off by exactly 1000x" without also catching coincidences


def _is_scaling_error(predicted_num: float, truth_num: float) -> bool:
    """True if predicted is truth off by ~1000x/1e6x/1e9x (or the inverse) --
    a unit-confusion pattern (e.g. reported in millions when the field is in
    thousands), as opposed to the model just getting the number wrong."""
    if truth_num == 0:
        return False
    ratio = predicted_num / truth_num
    for factor in _SCALING_FACTORS:
        if abs(ratio - factor) / factor <= _SCALING_TOLERANCE:
            return True
        if abs(ratio - (1 / factor)) / (1 / factor) <= _SCALING_TOLERANCE:
            return True
    return False


def classify_error(
    field_name: str,
    predicted,
    ground_truth,
    parse_stage: str | None = None,
) -> list[str]:
    """Return zero or more error tags explaining a field mismatch.

    A single mismatch can carry multiple tags (e.g. a value that's both
    numerically wrong AND only present because of parser recovery).
    """
    tags: list[str] = []

    truth_present = ground_truth is not None and str(ground_truth).strip() != ""
    pred_present = predicted is not None and str(predicted).strip() != ""

    if truth_present and not pred_present:
        tags.append("missing_required_field")
        return tags  # nothing else to classify -- there's no predicted value

    if not truth_present and pred_present:
        tags.append("hallucinated_field")
        return tags

    if not truth_present and not pred_present:
        return tags  # both correctly empty -- not an error

    # Both present but disagree.
    if field_name in NUMERIC_FIELDS:
        if fuzzy_numeric_match(predicted, ground_truth):
            return tags  # within tolerance -- not an error
        pred_num = parse_numeric_value(predicted)
        truth_num = parse_numeric_value(ground_truth)
        if pred_num is None or truth_num is None:
            tags.append("formatting_normalization_failure")
        elif _is_scaling_error(pred_num, truth_num):
            tags.append("numeric_scaling_failure")
        else:
            tags.append("wrong_field")
    else:
        if str(predicted).strip().lower() != str(ground_truth).strip().lower():
            tags.append("wrong_field")

    if tags and parse_stage in ("truncation_repair", "field_fallback"):
        tags.append("parser_recovery_failure")

    return tags
