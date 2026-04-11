"""Evaluation utilities: per-field accuracy, fuzzy financial matching, benchmarks."""

from evaluation.evaluate import (
    ALL_FIELDS,
    evaluate_dataset,
    evaluate_single,
    exact_match,
    fuzzy_financial_match,
    generate_sample_metrics,
)

__all__ = [
    "ALL_FIELDS",
    "evaluate_dataset",
    "evaluate_single",
    "exact_match",
    "fuzzy_financial_match",
    "generate_sample_metrics",
]
