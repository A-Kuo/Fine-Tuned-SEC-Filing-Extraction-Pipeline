"""Tests for src/core/monitoring_schemas.py, including the conversion
function against monitor.py's real dataclasses (not a hand-built fixture
that could drift from what to_dict() actually returns)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.monitor import check_accuracy_drift, check_latency_sla
from src.core.monitoring_schemas import (
    DriftReportModel,
    MonitoringReportModel,
    status_to_severity,
    to_monitoring_report_model,
)


class TestStatusToSeverity:
    def test_healthy_maps_to_info(self):
        assert status_to_severity("healthy") == "info"

    def test_warning_maps_to_warning(self):
        assert status_to_severity("warning") == "warning"

    def test_critical_maps_to_critical(self):
        assert status_to_severity("critical") == "critical"

    def test_unknown_status_defaults_to_info(self):
        assert status_to_severity("something_new") == "info"


class TestDriftReportModel:
    def test_schema_version_present(self):
        model = DriftReportModel(
            metric_name="accuracy", current_value=0.8, baseline_value=0.9,
            threshold=0.85, is_drifted=True,
        )
        assert model.schema_version == 1

    def test_optional_fields_default_none(self):
        model = DriftReportModel(
            metric_name="accuracy", current_value=0.8, baseline_value=0.9,
            threshold=0.85, is_drifted=True,
        )
        assert model.dataset_version is None
        assert model.z_score is None


class TestToMonitoringReportModel:
    def test_converts_real_drift_report_from_monitor_py(self):
        """Uses the actual check_accuracy_drift() output, not a hand-built
        dict shaped to match -- if to_dict()'s keys ever drift from what
        DriftReportModel expects, this test breaks for real."""
        drift = check_accuracy_drift(0.80, 0.94, 0.90, n_current=50, n_baseline=500)
        latency = check_latency_sla([100, 200, 300], sla_p99_ms=1200)

        model = to_monitoring_report_model(
            drift_reports=[drift], latency_report=latency,
            alerts=["CRITICAL: accuracy dropped"], status="critical",
            dataset_version="v1-test",
        )

        assert isinstance(model, MonitoringReportModel)
        assert model.severity == "critical"
        assert model.drift_reports[0].metric_name == "accuracy"
        assert model.drift_reports[0].is_drifted is True
        assert model.drift_reports[0].dataset_version == "v1-test"
        assert model.latency_report.p50_ms == latency.p50_ms
        assert model.dataset_version == "v1-test"

    def test_handles_no_latency_report(self):
        drift = check_accuracy_drift(0.94, 0.94, 0.90, n_current=50, n_baseline=500)
        model = to_monitoring_report_model(
            drift_reports=[drift], latency_report=None, alerts=[], status="healthy",
        )
        assert model.latency_report is None
        assert model.severity == "info"

    def test_report_id_is_unique_per_call(self):
        drift = check_accuracy_drift(0.94, 0.94, 0.90)
        m1 = to_monitoring_report_model([drift], None, [], "healthy")
        m2 = to_monitoring_report_model([drift], None, [], "healthy")
        assert m1.report_id != m2.report_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
