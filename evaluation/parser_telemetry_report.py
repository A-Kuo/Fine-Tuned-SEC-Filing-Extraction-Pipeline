"""Parser-stage telemetry report: how often does each of the 5 JSON-recovery
stages actually fire, and how much of schema-conformance is the parser
saving vs. the model getting it right directly?

Two corpora, reported separately so the two questions don't get conflated:
  - tests/fixtures/malformed_llm_outputs.jsonl: DELIBERATELY malformed
    outputs, one per stage, used to measure each stage's per-stage recovery
    rate against a controlled corpus (this is not "how often does this
    happen in production" -- it's "does each stage work at all, and which
    one recovers a given malformation").
  - A clean-output corpus (the same text, unmangled): measures how much
    stage 1 (direct parse) alone would handle if the model never needed
    help -- the model's own contribution, as a baseline to contrast against.

Usage:
    python evaluation/parser_telemetry_report.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.parser_telemetry import STAGES, ParseTelemetry
from src.extraction.postprocessing import parse_extraction

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "malformed_llm_outputs.jsonl"

# A small clean-output corpus (valid JSON, no malformation at all) --
# contrasts against the malformed corpus to isolate "model got it right
# unaided" from "parser saved it."
CLEAN_OUTPUT_CORPUS = [
    '{"company_name": "Acme Corp", "ticker": "ACME", "revenue": "1000000"}',
    '{"company_name": "Beta Inc", "filing_type": "10-K", "eps": "1.23"}',
    '{"company_name": "Gamma LLC", "date": "2024-01-01", "sector": "Tech"}',
]


def _load_fixtures() -> list[dict]:
    if not FIXTURE_PATH.exists():
        return []
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contribution_interpretation(malformed_recovery_rate: float | None, clean_direct_rate: float | None) -> str:
    parser_part = (
        f"the parser recovers {malformed_recovery_rate:.0%} of deliberately-malformed output"
        if malformed_recovery_rate is not None else "the parser's recovery rate is unavailable (empty fixture corpus)"
    )
    model_part = (
        f"stage 1 alone handles {clean_direct_rate:.0%} of well-formed output unaided"
        if clean_direct_rate is not None else "the model's unaided rate is unavailable (empty clean corpus)"
    )
    return (
        "Two different corpora, not directly subtractable into one number -- "
        f"read side by side: {parser_part} (stages 2-5 combined), while {model_part}."
    )


def run_report() -> dict:
    fixtures = _load_fixtures()

    per_case = []
    stage_recovery_counts = {stage: 0 for stage in STAGES}
    n_raised = 0

    for case in fixtures:
        telemetry = ParseTelemetry()
        raised = False
        try:
            parse_extraction(case["raw_output"], telemetry=telemetry)
        except Exception:
            raised = True
            n_raised += 1

        if telemetry.winning_stage:
            stage_recovery_counts[telemetry.winning_stage] += 1

        matches_expectation = (
            (raised and case.get("expected_raises"))
            or (not raised and telemetry.winning_stage == case.get("expected_stage"))
        )

        per_case.append({
            "name": case["name"],
            "expected_stage": case.get("expected_stage"),
            "expected_raises": case.get("expected_raises", False),
            "actual_winning_stage": telemetry.winning_stage,
            "actually_raised": raised,
            "matches_expectation": matches_expectation,
            "telemetry": telemetry.to_dict(),
        })

    malformed_recovered = sum(1 for c in per_case if not c["actually_raised"])
    malformed_total = len(per_case)
    malformed_recovery_rate = malformed_recovered / malformed_total if malformed_total else None

    # Clean-output baseline: how much stage 1 alone handles, unaided.
    clean_direct_hits = 0
    for text in CLEAN_OUTPUT_CORPUS:
        telemetry = ParseTelemetry()
        try:
            parse_extraction(text, telemetry=telemetry)
        except Exception:
            continue
        if telemetry.winning_stage == "direct":
            clean_direct_hits += 1
    clean_direct_rate = clean_direct_hits / len(CLEAN_OUTPUT_CORPUS) if CLEAN_OUTPUT_CORPUS else None

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "malformed_corpus numbers measure whether each parser STAGE works "
            "on a deliberately-malformed, hand-authored corpus -- not a "
            "production malformation rate (that requires real model output, "
            "not available without a GPU in this environment; see "
            "extraction_logs.parser_recovery_stage, added in "
            "db/migrations/0006_lineage.sql, for the real-traffic version of "
            "this report once live)."
        ),
        "malformed_corpus": {
            "n_cases": malformed_total,
            "recovered": malformed_recovered,
            "raised": n_raised,
            "recovery_rate": round(malformed_recovery_rate, 3) if malformed_recovery_rate is not None else None,
            "stage_recovery_counts": stage_recovery_counts,
            "all_expectations_matched": all(c["matches_expectation"] for c in per_case),
        },
        "clean_output_baseline": {
            "n_cases": len(CLEAN_OUTPUT_CORPUS),
            "direct_parse_hits": clean_direct_hits,
            "direct_parse_rate": round(clean_direct_rate, 3) if clean_direct_rate is not None else None,
            "interpretation": (
                "This is the model's own contribution: fraction of well-formed "
                "outputs stage 1 (direct parse) handles with zero parser help."
            ),
        },
        "parser_vs_model_contribution": {
            "parser_recovery_rate_on_malformed_corpus": round(malformed_recovery_rate, 3) if malformed_recovery_rate is not None else None,
            "model_direct_success_rate_on_clean_corpus": round(clean_direct_rate, 3) if clean_direct_rate is not None else None,
            "interpretation": _contribution_interpretation(malformed_recovery_rate, clean_direct_rate),
        },
        "cases": per_case,
    }


def main():
    results = run_report()

    out_dir = REPO_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parser_telemetry_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    mc = results["malformed_corpus"]
    cb = results["clean_output_baseline"]
    print(f"Malformed corpus: {mc['recovered']}/{mc['n_cases']} recovered ({mc['recovery_rate']:.0%})")
    print("  Stage recovery counts:")
    for stage, count in mc["stage_recovery_counts"].items():
        print(f"    {stage}: {count}")
    print(f"  All expectations matched: {mc['all_expectations_matched']}")
    print(f"\nClean-output baseline: {cb['direct_parse_hits']}/{cb['n_cases']} handled by direct parse alone ({cb['direct_parse_rate']:.0%})")
    print(f"\nReport written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
