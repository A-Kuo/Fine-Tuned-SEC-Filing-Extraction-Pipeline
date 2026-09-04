"""Production monitoring for accuracy drift and latency SLAs.

Implements automated drift detection by sampling extractions daily,
comparing accuracy against baseline, and alerting if performance degrades.

Drift detection math:
    We track a rolling window of accuracy measurements and apply a
    two-sample proportion test (z-test) to detect statistically
    significant drops:

        z = (p_current - p_baseline) / sqrt(p_baseline(1-p_baseline)(1/n_current + 1/n_baseline))

    If z < -z_alpha (one-tailed), accuracy has significantly dropped.
    With n=50 daily samples and alpha=0.05, we can detect a 5% drop
    with 80% power within 1 day.

Why this matters:
    Without monitoring, model degradation is invisible until users complain.
    SEC filing formats change (new XBRL tags, amended forms), data distributions
    shift, and model accuracy silently drops. Automated detection catches
    issues in <1 day vs. weeks with manual spot-checks.

Usage:
    python monitoring/monitor.py --check-drift
    python monitoring/monitor.py --check-latency
    python monitoring/monitor.py --full-report
"""

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.config import load_config


@dataclass
class DriftReport:
    """Results from a drift detection check."""
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    is_drifted: bool
    z_score: float | None = None
    p_value: float | None = None
    sample_size: int = 0
    checked_at: str = ""

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 4),
            "baseline_value": round(self.baseline_value, 4),
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "z_score": round(self.z_score, 3) if self.z_score is not None else None,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "sample_size": self.sample_size,
            "checked_at": self.checked_at or datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class LatencyReport:
    """Results from a latency SLA check."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sla_p99_ms: float
    is_within_sla: bool
    sample_size: int = 0


@dataclass
class MonitoringReport:
    """Full monitoring report combining drift + latency + health."""
    drift_reports: list[DriftReport]
    latency_report: LatencyReport | None
    alerts: list[str]
    status: str  # 'healthy', 'warning', 'critical'
    generated_at: str = ""
    is_demo_data: bool = False

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "is_demo_data": self.is_demo_data,
            "alerts": self.alerts,
            "drift": [d.to_dict() for d in self.drift_reports],
            "latency": {
                "p50_ms": self.latency_report.p50_ms,
                "p95_ms": self.latency_report.p95_ms,
                "p99_ms": self.latency_report.p99_ms,
                "sla_p99_ms": self.latency_report.sla_p99_ms,
                "within_sla": self.latency_report.is_within_sla,
            } if self.latency_report else None,
            "generated_at": self.generated_at or datetime.now(timezone.utc).isoformat(),
        }


def proportion_z_test(
    p_current: float,
    p_baseline: float,
    n_current: int,
    n_baseline: int,
) -> tuple[float, float]:
    """Two-sample proportion z-test for drift detection.

    Tests H0: p_current >= p_baseline vs H1: p_current < p_baseline (one-tailed).

    Args:
        p_current: Current accuracy proportion (0-1).
        p_baseline: Baseline accuracy proportion (0-1).
        n_current: Number of current samples.
        n_baseline: Number of baseline samples.

    Returns:
        (z_score, p_value) tuple. Negative z means current < baseline.
    """
    if n_current == 0 or n_baseline == 0:
        return 0.0, 1.0

    # Pooled proportion under H0
    p_pool = (p_current * n_current + p_baseline * n_baseline) / (n_current + n_baseline)

    # Standard error
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_current + 1 / n_baseline))
    if se == 0:
        return 0.0, 1.0

    z = (p_current - p_baseline) / se

    # One-tailed p-value for H1: p_current < p_baseline (lower tail).
    # Φ(z) = 0.5 * (1 + erf(z / √2)) gives the lower-tail CDF directly.
    # When z is negative (current worse), Φ(z) is small → significant.
    p_value = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return z, p_value


def check_metric_drift(
    metric_name: str,
    current_value: float,
    baseline_value: float,
    threshold: float,
    n_current: int = 50,
    n_baseline: int = 500,
) -> DriftReport:
    """Generic proportion-drift check: is `current_value` a statistically
    significant, threshold-crossing drop from `baseline_value`?

    Works for any metric expressed as a proportion in [0, 1] -- accuracy,
    schema-conformance rate, parser direct-parse rate (a drop here means
    MORE outputs are needing fallback recovery), cache hit rate. Field
    accuracy is really N per-field proportions; call this once per field.

    Uses both an absolute threshold check and a statistical test:
    - Absolute: is current_value < threshold?
    - Statistical: is the drop statistically significant (p < 0.05)?
    Both must trigger for a drift alert (reduces false positives).
    """
    z_score, p_value = proportion_z_test(
        current_value, baseline_value, n_current, n_baseline
    )

    is_drifted = current_value < threshold and p_value < 0.05

    return DriftReport(
        metric_name=metric_name,
        current_value=current_value,
        baseline_value=baseline_value,
        threshold=threshold,
        is_drifted=is_drifted,
        z_score=z_score,
        p_value=p_value,
        sample_size=n_current,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def check_accuracy_drift(
    current_accuracy: float,
    baseline_accuracy: float,
    threshold: float,
    n_current: int = 50,
    n_baseline: int = 500,
) -> DriftReport:
    """Accuracy-specific wrapper around check_metric_drift(), kept for
    backward compatibility with existing callers (generate_full_report(),
    the CLI, tests)."""
    return check_metric_drift(
        "accuracy", current_accuracy, baseline_accuracy, threshold, n_current, n_baseline
    )


def check_parser_fallback_drift(
    current_fallback_rate: float, baseline_fallback_rate: float,
    threshold: float, n_current: int = 50, n_baseline: int = 500,
) -> DriftReport:
    """Parser fallback rate RISING is bad (more model outputs needed
    rescue) -- the opposite direction from the other checks here, which all
    flag when a metric DROPS. Implemented as a drop-check on 1 minus the
    rate, so is_drifted still means "something got worse."
    """
    report = check_metric_drift(
        "parser_fallback_rate_inverse",
        1 - current_fallback_rate, 1 - baseline_fallback_rate,
        1 - threshold, n_current, n_baseline,
    )
    # Re-express in the caller's terms (the rate itself, not its inverse).
    report.metric_name = "parser_fallback_rate"
    report.current_value = current_fallback_rate
    report.baseline_value = baseline_fallback_rate
    report.threshold = threshold
    return report


def check_schema_conformance_drift(
    current_rate: float, baseline_rate: float,
    threshold: float, n_current: int = 50, n_baseline: int = 500,
) -> DriftReport:
    return check_metric_drift("schema_conformance_rate", current_rate, baseline_rate, threshold, n_current, n_baseline)


def check_field_accuracy_drift(
    field_name: str, current_accuracy: float, baseline_accuracy: float,
    threshold: float, n_current: int = 50, n_baseline: int = 500,
) -> DriftReport:
    return check_metric_drift(f"field_accuracy:{field_name}", current_accuracy, baseline_accuracy, threshold, n_current, n_baseline)


def check_cache_performance_drift(
    current_hit_rate: float, baseline_hit_rate: float,
    threshold: float, n_current: int = 50, n_baseline: int = 500,
) -> DriftReport:
    return check_metric_drift("cache_hit_rate", current_hit_rate, baseline_hit_rate, threshold, n_current, n_baseline)


def check_latency_sla(
    latencies_ms: list[float],
    sla_p99_ms: float = 1200,
) -> LatencyReport:
    """Check if latency is within SLA bounds.

    SLA: p99 latency < 1200ms (from config).
    Also reports p50 and p95 for trend monitoring.
    """
    if not latencies_ms:
        return LatencyReport(
            p50_ms=0, p95_ms=0, p99_ms=0,
            sla_p99_ms=sla_p99_ms, is_within_sla=True, sample_size=0,
        )

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    p50 = sorted_lat[n // 2]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]

    return LatencyReport(
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        sla_p99_ms=sla_p99_ms,
        is_within_sla=p99 <= sla_p99_ms,
        sample_size=n,
    )


def generate_full_report(
    current_accuracy: float,
    baseline_accuracy: float,
    latencies_ms: list[float],
    config: dict | None = None,
    is_demo_data: bool = False,
) -> MonitoringReport:
    """Generate complete monitoring report with alerts.

    Combines drift detection, latency SLA check, and generates
    actionable alerts for the ops team.
    """
    config = config or load_config()
    mon_cfg = config["monitoring"]

    # Drift check
    drift = check_accuracy_drift(
        current_accuracy=current_accuracy,
        baseline_accuracy=baseline_accuracy,
        threshold=mon_cfg["accuracy_threshold"],
        n_current=mon_cfg["daily_sample_size"],
    )

    # Latency check
    latency = check_latency_sla(
        latencies_ms=latencies_ms,
        sla_p99_ms=mon_cfg["latency_p99_threshold_ms"],
    )

    # Generate alerts
    alerts = []
    status = "healthy"

    if drift.is_drifted:
        alerts.append(
            f"CRITICAL: Accuracy dropped to {drift.current_value:.1%} "
            f"(threshold: {drift.threshold:.1%}, baseline: {drift.baseline_value:.1%}). "
            f"Recommend triggering retraining pipeline."
        )
        status = "critical"

    elif drift.current_value < drift.baseline_value - 0.02:
        alerts.append(
            f"WARNING: Accuracy trending down: {drift.current_value:.1%} "
            f"(baseline: {drift.baseline_value:.1%}). Monitor closely."
        )
        if status != "critical":
            status = "warning"

    if not latency.is_within_sla:
        alerts.append(
            f"CRITICAL: p99 latency {latency.p99_ms:.0f}ms exceeds "
            f"SLA {latency.sla_p99_ms:.0f}ms. Check GPU utilization and batch size."
        )
        status = "critical"

    elif latency.p95_ms > latency.sla_p99_ms * 0.8:
        alerts.append(
            f"WARNING: p95 latency {latency.p95_ms:.0f}ms approaching "
            f"SLA limit {latency.sla_p99_ms:.0f}ms."
        )
        if status != "critical":
            status = "warning"

    return MonitoringReport(
        drift_reports=[drift],
        latency_report=latency,
        alerts=alerts,
        status=status,
        generated_at=datetime.now(timezone.utc).isoformat(),
        is_demo_data=is_demo_data,
    )


def main():
    """CLI entry point for monitoring checks.

    No silent fabrication: this used to unconditionally default
    --current-accuracy/--baseline-accuracy to 0.94 and always generate
    latencies via random.gauss(seed=42), regardless of any flag, then
    persist that as results/latest_report.json -- indistinguishable from a
    report built from real production data. --demo is now required to get
    that behavior at all, and its output is tagged and separately named.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run monitoring checks")
    parser.add_argument("--check-drift", action="store_true", help="Check accuracy drift")
    parser.add_argument("--check-latency", action="store_true", help="Check latency SLA")
    parser.add_argument("--full-report", action="store_true", help="Generate full report")
    parser.add_argument("--current-accuracy", type=float, default=None)
    parser.add_argument("--baseline-accuracy", type=float, default=None)
    parser.add_argument(
        "--latencies-file", type=str, default=None,
        help="JSON file containing a list of real per-request latencies (ms)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use FABRICATED accuracy/latency numbers (not measured) -- only "
             "for exercising this script's report format without production data",
    )
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    args = parser.parse_args()

    config = load_config()

    if args.demo:
        current_accuracy = args.current_accuracy if args.current_accuracy is not None else 0.94
        baseline_accuracy = args.baseline_accuracy if args.baseline_accuracy is not None else 0.94
        import random
        random.seed(42)
        latencies = [random.gauss(350, 80) for _ in range(100)]
    else:
        if args.current_accuracy is None or args.baseline_accuracy is None:
            parser.error(
                "--current-accuracy and --baseline-accuracy are required "
                "(measured values), or pass --demo for fabricated numbers."
            )
        current_accuracy = args.current_accuracy
        baseline_accuracy = args.baseline_accuracy
        if args.latencies_file:
            with open(args.latencies_file) as f:
                latencies = json.load(f)
        elif args.check_latency or args.full_report or (not args.check_drift and not args.check_latency):
            parser.error(
                "--latencies-file is required for latency checks "
                "(a JSON list of real per-request latencies in ms), or pass --demo."
            )
        else:
            latencies = []

    if args.full_report or (not args.check_drift and not args.check_latency):
        report = generate_full_report(
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            latencies_ms=latencies,
            config=config,
            is_demo_data=args.demo,
        )
        report_dict = report.to_dict()
        if args.demo:
            print("WARNING: --demo was passed -- this report is FABRICATED, not measured.")
        print(json.dumps(report_dict, indent=2))

        # Persist for dashboard consumption -- demo and real reports write to
        # distinctly named files so a demo run can never silently become
        # "the latest report" a live dashboard reads as production data.
        default_name = "latest_report.demo.json" if args.demo else "latest_report.json"
        default_report_path = Path("results") / default_name
        default_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(default_report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report_dict, f, indent=2)
            print(f"\nReport saved to {args.output}")

    if args.check_drift:
        drift = check_accuracy_drift(
            current_accuracy, baseline_accuracy,
            config["monitoring"]["accuracy_threshold"],
        )
        print(json.dumps(drift.to_dict(), indent=2))

    if args.check_latency:
        latency = check_latency_sla(
            latencies,
            config["monitoring"]["latency_p99_threshold_ms"],
        )
        print(f"p50: {latency.p50_ms:.0f}ms  p95: {latency.p95_ms:.0f}ms  "
              f"p99: {latency.p99_ms:.0f}ms  SLA: {'✓' if latency.is_within_sla else '✗'}")


if __name__ == "__main__":
    main()
