"""Alert system for monitoring threshold breaches.

Sends notifications when accuracy drops or latency exceeds SLA.
Supports multiple backends: console logging (always), email (optional),
and webhook (optional for Slack/PagerDuty integration).

In production, this would integrate with PagerDuty or OpsGenie.
For the prototype, we log alerts and write to a file for review.
"""

import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger

from monitoring.monitor import MonitoringReport


ALERT_LOG_PATH = Path("results/alerts.jsonl")


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

        # Email if configured
        if config and config.get("monitoring", {}).get("alert_email"):
            _send_email_alert(
                to_addr=config["monitoring"]["alert_email"],
                subject=f"[Financial LLM] {report.status.upper()}: {alert_msg[:50]}...",
                body=_format_email_body(report, alert_msg),
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
