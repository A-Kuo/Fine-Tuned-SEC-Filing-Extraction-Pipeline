"""Evaluation metrics beyond overall exact-match accuracy: per-field
accuracy, schema-conformance rate, exact-JSON-match rate, fuzzy numeric
match rate, and null-handling correctness.

Confidence calibration is deliberately NOT implemented here: confidence in
this pipeline is a heuristic constant assigned per extraction method
(see MetricRecord.confidence in src/core/schemas.py), not a model-emitted
probability. A calibration curve computed against a non-probabilistic
heuristic score would itself be a fabricated-looking number -- see
report_confidence_calibration() below for what this returns instead.
"""

from __future__ import annotations

from typing import Any

from evaluation.error_taxonomy import NUMERIC_FIELDS, classify_error
from src.extraction.numeric_normalize import fuzzy_numeric_match

TEXT_FIELDS = [
    "filing_id", "company_name", "ticker", "filing_type", "date",
    "fiscal_year_end", "sector",
]
ALL_FIELDS = TEXT_FIELDS + list(NUMERIC_FIELDS)


def per_field_accuracy(predictions: list[dict], ground_truths: list[dict]) -> dict[str, dict]:
    """For each field: fraction correct (exact match for text fields, fuzzy
    numeric match for numeric fields), with correct/total counts."""
    results = {field: {"correct": 0, "total": 0} for field in ALL_FIELDS}

    for pred, truth in zip(predictions, ground_truths):
        for field in ALL_FIELDS:
            pred_val = pred.get(field)
            truth_val = truth.get(field)
            if truth_val is None and pred_val is None:
                continue  # both correctly absent -- not counted either way
            results[field]["total"] += 1
            if field in NUMERIC_FIELDS:
                correct = fuzzy_numeric_match(pred_val, truth_val)
            else:
                correct = (
                    pred_val is not None and truth_val is not None
                    and str(pred_val).strip().lower() == str(truth_val).strip().lower()
                )
            if correct:
                results[field]["correct"] += 1

    for field, counts in results.items():
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else None

    return results


def exact_json_match_rate(predictions: list[dict], ground_truths: list[dict]) -> float:
    """Fraction of predictions where every field matches ground truth
    (fuzzy for numerics, exact for text) -- the strictest metric."""
    if not predictions:
        return 0.0

    n_exact = 0
    for pred, truth in zip(predictions, ground_truths):
        all_match = True
        for field in ALL_FIELDS:
            pred_val, truth_val = pred.get(field), truth.get(field)
            if truth_val is None and pred_val is None:
                continue
            if field in NUMERIC_FIELDS:
                if not fuzzy_numeric_match(pred_val, truth_val):
                    all_match = False
                    break
            elif pred_val is None or str(pred_val).strip().lower() != str(truth_val).strip().lower():
                all_match = False
                break
        if all_match:
            n_exact += 1

    return n_exact / len(predictions)


def null_handling_correctness(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """How often the model correctly abstains (predicts null when ground
    truth is null) vs. incorrectly abstains or hallucinates."""
    correct_abstain = 0
    incorrect_abstain = 0  # truth present, predicted null
    hallucinated = 0  # truth null, predicted present
    total = 0

    for pred, truth in zip(predictions, ground_truths):
        for field in ALL_FIELDS:
            pred_val, truth_val = pred.get(field), truth.get(field)
            truth_present = truth_val is not None and str(truth_val).strip() != ""
            pred_present = pred_val is not None and str(pred_val).strip() != ""
            total += 1
            if not truth_present and not pred_present:
                correct_abstain += 1
            elif truth_present and not pred_present:
                incorrect_abstain += 1
            elif not truth_present and pred_present:
                hallucinated += 1

    return {
        "total_field_checks": total,
        "correct_abstain": correct_abstain,
        "incorrect_abstain_rate": incorrect_abstain / total if total else None,
        "hallucination_rate": hallucinated / total if total else None,
    }


def error_taxonomy_breakdown(
    predictions: list[dict],
    ground_truths: list[dict],
    parse_stages: list[str | None] | None = None,
) -> dict[str, int]:
    """Aggregate error_taxonomy.classify_error() tags across a whole eval run."""
    counts: dict[str, int] = {}
    stages = parse_stages or [None] * len(predictions)

    for pred, truth, stage in zip(predictions, ground_truths, stages):
        for field in ALL_FIELDS:
            for tag in classify_error(field, pred.get(field), truth.get(field), parse_stage=stage):
                counts[tag] = counts.get(tag, 0) + 1

    return counts


def report_confidence_calibration(predictions: list[dict]) -> dict[str, Any]:
    """Confidence calibration is NOT applicable to this pipeline today.

    Confidence in src/core/schemas.py's MetricRecord is a fixed heuristic
    constant per extraction method (xbrl/heuristic/llm), not a model-emitted
    probability -- there is nothing to calibrate a reliability curve against.
    Computing one anyway would produce a number that looks measured but
    isn't, exactly the pattern flagged in docs/TRUTH_AUDIT.md. This function
    documents that explicitly rather than fabricating a curve.
    """
    return {
        "applicable": False,
        "reason": (
            "Confidence is a fixed heuristic constant per extraction method "
            "(see MetricRecord.confidence in src/core/schemas.py), not a "
            "model-emitted probability. A calibration curve requires the "
            "latter -- e.g. deriving per-field confidence from average "
            "token log-probability over the field's generated JSON span, "
            "which this pipeline does not currently do."
        ),
    }
