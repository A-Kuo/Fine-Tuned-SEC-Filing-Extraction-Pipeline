"""Retention-friendly, versioned monitoring report schemas.

monitoring/monitor.py's DriftReport/LatencyReport/MonitoringReport are plain
dataclasses with no schema_version and no shared severity vocabulary --
monitoring/alerts.py's dispatch logic uses ad hoc strings instead. These
Pydantic mirrors add report_id, lineage fields (dataset_version, matching
src/core/dataset_manifest.py's convention), and schema_version so a report
saved today stays parseable as the schema evolves -- same pattern as
FilingRecord in src/core/schemas.py.

These are NEW schemas alongside the dataclasses in monitor.py, not a
replacement -- monitor.py's own report objects remain the working
implementation; call to_monitoring_report_model() to get the versioned,
serializable form when persisting a report for retention.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "warning", "critical"]


class DriftReportModel(BaseModel):
    schema_version: int = 1
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    is_drifted: bool
    z_score: float | None = None
    p_value: float | None = None
    sample_size: int = 0
    dataset_version: str | None = None
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LatencyReportModel(BaseModel):
    schema_version: int = 1
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sla_p99_ms: float
    is_within_sla: bool
    sample_size: int = 0
    dataset_version: str | None = None


class MonitoringReportModel(BaseModel):
    schema_version: int = 1
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    drift_reports: list[DriftReportModel] = Field(default_factory=list)
    latency_report: LatencyReportModel | None = None
    alerts: list[str] = Field(default_factory=list)
    severity: Severity = "info"
    dataset_version: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def status_to_severity(status: str) -> Severity:
    """Maps monitor.py's existing status vocabulary ("healthy"/"warning"/
    "critical") onto the shared Severity type, so alerts.py and
    MonitoringReportModel can share one vocabulary instead of two ad hoc
    string sets that could drift apart."""
    mapping: dict[str, Severity] = {
        "healthy": "info",
        "warning": "warning",
        "critical": "critical",
    }
    return mapping.get(status, "info")


def to_monitoring_report_model(
    drift_reports: list,
    latency_report,
    alerts: list[str],
    status: str,
    dataset_version: str | None = None,
) -> MonitoringReportModel:
    """Converts monitor.py's dataclass-based DriftReport/LatencyReport
    objects (via their existing .to_dict()) into the versioned Pydantic
    form, for persisting a retention-friendly report."""
    return MonitoringReportModel(
        drift_reports=[DriftReportModel(**d.to_dict(), dataset_version=dataset_version) for d in drift_reports],
        latency_report=(
            LatencyReportModel(
                p50_ms=latency_report.p50_ms,
                p95_ms=latency_report.p95_ms,
                p99_ms=latency_report.p99_ms,
                sla_p99_ms=latency_report.sla_p99_ms,
                is_within_sla=latency_report.is_within_sla,
                sample_size=latency_report.sample_size,
                dataset_version=dataset_version,
            )
            if latency_report else None
        ),
        alerts=alerts,
        severity=status_to_severity(status),
        dataset_version=dataset_version,
    )
