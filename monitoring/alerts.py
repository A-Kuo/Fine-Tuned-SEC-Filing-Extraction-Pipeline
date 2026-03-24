"""Alert system for monitoring threshold breaches.

Sends notifications when accuracy drops or latency exceeds SLA.
Supports multiple backends:
  - Console logging (always)
  - File append (always)
  - Email (optional, configure monitoring.alert_email)
  - Prometheus Alertmanager (optional, configure monitoring.alertmanager_url)
  - Slack webhook (optional, configure monitoring.slack_webhook_url)
"""

import json
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from loguru import logger

from monitoring.monitor import MonitoringReport


ALERT_LOG_PATH = Path("results/alerts.jsonl")

# Alertmanager severity map: our status → AM severity label
_SEVERITY_MAP = {
    "healthy": "info",
    "warning": "warning",
    "critical": "critical",
    "degraded": "warning",
}


def send_alerts(report: MonitoringReport, config: dict | None = None) -> int:
    """Process and dispatch all alerts from a monitoring report.

    Returns number of alerts sent.
    """
    if not report.alerts:
        return 0

    alert_count = 0
    for alert_msg in report.alerts:
        alert_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": report.status,
            "message": alert_msg,
        }

        # Always log
        if "CRITICAL" in alert_msg:
            logger.critical(alert_msg)
        else:
            logger.warning(alert_msg)

        # Persist to file
        _log_alert(alert_record)

        if config:
            mon_cfg = config.get("monitoring", {})

            # Email
            if mon_cfg.get("alert_email"):
                _send_email_alert(
                    to_addr=mon_cfg["alert_email"],
                    subject=f"[Financial LLM] {report.status.upper()}: {alert_msg[:50]}...",
                    body=_format_email_body(report, alert_msg),
                )

            # Prometheus Alertmanager
            if mon_cfg.get("alertmanager_url"):
                _send_alertmanager(
                    url=mon_cfg["alertmanager_url"],
                    alert_msg=alert_msg,
                    status=report.status,
                    report=report,
                )

            # Slack
            if mon_cfg.get("slack_webhook_url"):
                _send_slack(
                    url=mon_cfg["slack_webhook_url"],
                    alert_msg=alert_msg,
                    status=report.status,
                )

        alert_count += 1

    return alert_count


def _log_alert(record: dict) -> None:
    """Append alert to persistent log file."""
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _format_email_body(report: MonitoringReport, alert_msg: str) -> str:
    """Format alert as human-readable email body."""
    body = f"""Financial LLM Monitoring Alert
{'=' * 40}

Status: {report.status.upper()}
Time: {report.generated_at}

Alert: {alert_msg}

"""
    if report.drift_reports:
        drift = report.drift_reports[0]
        body += f"""Drift Detection:
  Current Accuracy: {drift.current_value:.1%}
  Baseline:         {drift.baseline_value:.1%}
  Threshold:        {drift.threshold:.1%}
  Z-Score:          {drift.z_score:.3f}
  Sample Size:      {drift.sample_size}

"""

    if report.latency_report:
        lat = report.latency_report
        body += f"""Latency:
  p50:  {lat.p50_ms:.0f}ms
  p95:  {lat.p95_ms:.0f}ms
  p99:  {lat.p99_ms:.0f}ms
  SLA:  {lat.sla_p99_ms:.0f}ms ({'PASS' if lat.is_within_sla else 'FAIL'})

"""

    body += """Action Required:
  - If accuracy dropped: trigger retraining pipeline
  - If latency spiked: check GPU utilization, reduce batch size
  - Review recent extraction logs for anomalies
"""
    return body


def _send_alertmanager(url: str, alert_msg: str, status: str, report: MonitoringReport) -> bool:
    """POST a firing alert to Prometheus Alertmanager /api/v2/alerts.

    Alertmanager expects a list of alert objects. We send one per call.
    """
    try:
        import httpx

        now = datetime.now(timezone.utc).isoformat()
        severity = _SEVERITY_MAP.get(status, "warning")
        payload = [
            {
                "labels": {
                    "alertname": "FinDocMonitorAlert",
                    "severity": severity,
                    "service": "findoc-analyzer",
                    "status": status,
                },
                "annotations": {
                    "summary": alert_msg[:100],
                    "description": alert_msg,
                    "generated_at": report.generated_at,
                },
                "startsAt": now,
            }
        ]
        am_url = url.rstrip("/") + "/api/v2/alerts"
        r = httpx.post(am_url, json=payload, timeout=10)
        if r.status_code < 300:
            logger.info(f"Alert posted to Alertmanager ({r.status_code})")
            return True
        logger.warning(f"Alertmanager returned {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.warning(f"Alertmanager dispatch failed (non-critical): {e}")
    return False


def _send_slack(url: str, alert_msg: str, status: str) -> bool:
    """POST a Slack-compatible webhook message."""
    try:
        import httpx

        severity = _SEVERITY_MAP.get(status, "warning")
        color = {"critical": "#D32F2F", "warning": "#F9A825", "info": "#1976D2"}.get(severity, "#9E9E9E")
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"[FinDocAnalyzer] {status.upper()}",
                    "text": alert_msg,
                    "footer": "FinDocAnalyzer monitoring",
                    "ts": int(datetime.utcnow().timestamp()),
                }
            ]
        }
        r = httpx.post(url, json=payload, timeout=10)
        if r.status_code < 300:
            logger.info("Slack alert sent")
            return True
        logger.warning(f"Slack webhook returned {r.status_code}")
    except Exception as e:
        logger.warning(f"Slack alert failed (non-critical): {e}")
    return False


def _send_email_alert(to_addr: str, subject: str, body: str) -> bool:
    """Send email alert. Returns True on success."""
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = "financial-llm-monitor@noreply.local"
        msg["To"] = to_addr

        with smtplib.SMTP("localhost", 25, timeout=10) as server:
            server.send_message(msg)
        logger.info(f"Alert email sent to {to_addr}")
        return True
    except Exception as e:
        logger.warning(f"Email alert failed (non-critical): {e}")
        return False
