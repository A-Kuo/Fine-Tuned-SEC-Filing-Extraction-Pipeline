"""Synthetic incident walkthrough: degraded input distribution -> triggered
alert -> saved report -> recommended remediation.

Real measurements, not fabricated before/after numbers: "baseline" is the
parser's direct-parse rate on evaluation/parser_telemetry_report.py's
CLEAN_OUTPUT_CORPUS (well-formed model output, no malformation); "degraded"
is the fallback rate on tests/fixtures/malformed_llm_outputs.jsonl
(deliberately malformed output). Both are genuinely run through
parse_extraction(), not hardcoded. This demonstrates the mechanism (real
input -> real telemetry -> real drift check -> real alert), not a claim that
this specific corpus represents actual production traffic -- production
traffic requires a GPU and live logs this environment doesn't have (see
extraction_logs.parser_recovery_stage, added in
db/migrations/0006_lineage.sql, for how that would work once live).

Usage:
    python scripts/simulate_incident.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from monitoring.alerts import send_alerts
from monitoring.monitor import MonitoringReport, check_parser_fallback_drift
from src.core.monitoring_schemas import to_monitoring_report_model
from src.extraction.parser_telemetry import ParseTelemetry
from src.extraction.postprocessing import parse_extraction

CLEAN_OUTPUT_CORPUS = [
    '{"company_name": "Acme Corp", "ticker": "ACME", "revenue": "1000000"}',
    '{"company_name": "Beta Inc", "filing_type": "10-K", "eps": "1.23"}',
    '{"company_name": "Gamma LLC", "date": "2024-01-01", "sector": "Tech"}',
]


def _measure_fallback_rate(corpus: list[str]) -> tuple[float, int]:
    """Fraction of the corpus that needed stage 2-5 recovery (didn't parse
    directly). Returns (rate, n)."""
    n_needed_fallback = 0
    for text in corpus:
        telemetry = ParseTelemetry()
        try:
            parse_extraction(text, telemetry=telemetry)
        except Exception:
            n_needed_fallback += 1  # total failure also counts as "needed help and didn't get it"
            continue
        if telemetry.winning_stage != "direct":
            n_needed_fallback += 1
    return n_needed_fallback / len(corpus) if corpus else 0.0, len(corpus)


def load_malformed_corpus() -> list[str]:
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "malformed_llm_outputs.jsonl"
    if not fixture_path.exists():
        return []
    cases = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c["raw_output"] for c in cases]


def main():
    baseline_rate, n_baseline = _measure_fallback_rate(CLEAN_OUTPUT_CORPUS)
    degraded_corpus = load_malformed_corpus()
    current_rate, n_current = _measure_fallback_rate(degraded_corpus)

    print(f"Baseline (clean output corpus, n={n_baseline}): fallback rate = {baseline_rate:.1%}")
    print(f"Degraded (malformed output corpus, n={n_current}): fallback rate = {current_rate:.1%}")

    drift = check_parser_fallback_drift(
        current_fallback_rate=current_rate,
        baseline_fallback_rate=baseline_rate,
        threshold=0.20,
        n_current=n_current,
        n_baseline=n_baseline,
    )

    alerts = []
    status = "healthy"
    if drift.is_drifted:
        alerts.append(
            f"CRITICAL: parser fallback rate rose to {drift.current_value:.1%} "
            f"(baseline: {drift.baseline_value:.1%}, threshold: {drift.threshold:.1%}). "
            f"z={drift.z_score:.2f}, p={drift.p_value:.4f}."
        )
        status = "critical"

    print(f"\nDrift check: is_drifted={drift.is_drifted}  z={drift.z_score:.3f}  p={drift.p_value:.4f}")
    print(f"Status: {status}")

    report = MonitoringReport(drift_reports=[drift], latency_report=None, alerts=alerts, status=status)

    # config=None -> send_alerts() logs to results/alerts.jsonl only, no
    # Slack/email/Alertmanager -- safe to run in any environment.
    n_sent = send_alerts(report, config=None)
    print(f"\nAlerts dispatched (local file sink only): {n_sent}")

    versioned = to_monitoring_report_model(
        drift_reports=[drift], latency_report=None, alerts=alerts, status=status,
        dataset_version="incident-simulation-v1",
    )
    out_path = REPO_ROOT / "evaluation" / "results" / "incident_simulation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(versioned.model_dump_json(indent=2), encoding="utf-8")
    print(f"Report written to {out_path.relative_to(REPO_ROOT)}")

    print("\n--- Remediation runbook ---")
    if drift.is_drifted:
        print("1. Run `python evaluation/parser_telemetry_report.py` to see which specific")
        print("   stage(s) are absorbing the increased malformation and their reason codes.")
        print("2. Inspect the reason_code breakdown for a concentrated failure mode (e.g. many")
        print("   'no_json_start_or_brace_mismatch_too_large' entries suggests truncation from")
        print("   a max_tokens setting that's too low for the current prompt/output shape).")
        print("3. If the pattern is a new, previously-unseen malformation, add a fixture case to")
        print("   tests/fixtures/malformed_llm_outputs.jsonl and extend the relevant stage.")
        print("4. Re-run this script after the fix; is_drifted should return to False.")
    else:
        print("No drift detected against this corpus -- nothing to remediate. Re-run after")
        print("substituting real production output (once available) for a live measurement.")


if __name__ == "__main__":
    main()
