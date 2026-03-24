"""Streamlit Monitoring Dashboard.

Loads metrics from PostgreSQL when available; falls back to demo data.
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config


def generate_demo_data():
    """Fallback demo data when database is unavailable."""
    random.seed(42)
    now = datetime.utcnow()

    accuracy_history = []
    for i in range(30):
        day = now - timedelta(days=29 - i)
        acc = 0.94 + random.gauss(0, 0.008)
        accuracy_history.append({
            "date": day.strftime("%Y-%m-%d"),
            "accuracy": max(0.88, min(0.98, acc)),
            "sample_size": random.randint(40, 60),
        })

    latencies = [max(100, random.gauss(350, 80)) for _ in range(500)]
    statuses = random.choices(
        ["success", "validation_error", "parse_error", "timeout"],
        weights=[94, 3, 2, 1],
        k=100,
    )
    cache_stats = {
        "hits": 9823,
        "misses": 177,
        "hit_rate": 0.982,
        "used_memory_mb": 48.3,
    }
    return accuracy_history, latencies, statuses, cache_stats


@st.cache_data(ttl=300)

def _load_dashboard_data_cached(days: int):
    """Load time series and logs from PostgreSQL via DatabaseManager."""
    try:
        from src.database import DatabaseManager

        db = DatabaseManager.from_config()
        accuracy_history = db.get_daily_extraction_counts(days=days)
        if not accuracy_history:
            raise ValueError("empty")
        storage_stats = db.get_stats().get("storage") or {}
        latencies = []
        logs = db.get_recent_extraction_logs(500)
        for row in logs:
            if row.get("latency_ms") is not None:
                try:
                    latencies.append(float(row["latency_ms"]))
                except (TypeError, ValueError):
                    pass
        if not latencies:
            latencies = [storage_stats.get("latency_p50_ms", 300) or 300]

        statuses = [row.get("status", "unknown") for row in logs[:100]]
        if not statuses:
            statuses = ["success"]

        cache_stats = db.cache.get_stats()
        if not cache_stats.get("available"):
            cache_stats = {"hit_rate": 0, "hits": 0, "misses": 0, "used_memory_mb": 0}
        db.close()
        return accuracy_history, latencies, statuses, cache_stats
    except Exception:
        return generate_demo_data()


def load_dashboard_data(days: int = 30):
    """Public wrapper with 5-minute cache for expensive DB aggregations."""
    return _load_dashboard_data_cached(days)


@st.cache_data(ttl=300)
def _recent_extractions_cached():
    try:
        from src.database import DatabaseManager

        db = DatabaseManager.from_config()
        rows = db.get_recent_extractions_dashboard(20)
        db.close()
        return rows
    except Exception:
        return []


def _load_latest_report() -> dict | None:
    """Load the most recent monitoring report written by monitor.py."""
    report_path = Path("results/latest_report.json")
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except Exception:
        return None


def _render_drift_panel(report: dict) -> None:
    """Render a drift + latency status panel from a MonitoringReport dict."""
    import plotly.graph_objects as go

    status = report.get("status", "unknown")
    color = {"healthy": "green", "warning": "orange", "critical": "red"}.get(status, "grey")

    st.subheader("Drift & SLA Status")
    st.markdown(
        f"<span style='color:{color};font-size:1.2em;font-weight:bold;'>"
        f"● {status.upper()}</span> &nbsp; "
        f"<span style='color:grey;font-size:0.85em;'>checked at "
        f"{report.get('generated_at','')[:16]}</span>",
        unsafe_allow_html=True,
    )

    drift_list = report.get("drift", [])
    lat = report.get("latency")

    col_d, col_l = st.columns(2)
    with col_d:
        if drift_list:
            d = drift_list[0]
            st.metric(
                "Accuracy (current vs baseline)",
                f"{d['current_value']:.1%}",
                delta=f"{d['current_value'] - d['baseline_value']:+.1%}",
                delta_color="normal" if not d["is_drifted"] else "inverse",
            )
            z = d.get("z_score")
            if z is not None:
                st.caption(f"z-score: {z:.2f}  |  threshold: {d['threshold']:.0%}")
            st.caption("🔴 Drift detected" if d["is_drifted"] else "✅ No drift")
        else:
            st.info("No drift report available.")

    with col_l:
        if lat:
            sla_ok = lat.get("within_sla", True)
            st.metric(
                "p99 Latency",
                f"{lat['p99_ms']:.0f}ms",
                delta=f"SLA {lat['sla_p99_ms']:.0f}ms",
                delta_color="off",
            )
            st.caption(
                f"p50: {lat['p50_ms']:.0f}ms  p95: {lat['p95_ms']:.0f}ms  "
                + ("✅ Within SLA" if sla_ok else "🔴 SLA breached")
            )
        else:
            st.info("No latency report available.")

    alerts = report.get("alerts", [])
    if alerts:
        for a in alerts:
            if "CRITICAL" in a:
                st.error(a)
            else:
                st.warning(a)


def main():
    st.set_page_config(
        page_title="Financial LLM Monitor",
        page_icon="📊",
        layout="wide",
    )
    st_autorefresh(interval=60 * 1000, key="monitor_refresh")

    st.title("📊 Financial LLM Extraction Monitor")
    st.caption("Real-time monitoring for SEC filing extraction system")

    config = load_config()
    days = st.sidebar.selectbox("Time range (days)", [7, 14, 30], index=2)
    accuracy_history, latencies, statuses, cache_stats = load_dashboard_data(days=days)

    col1, col2, col3, col4 = st.columns(4)

    if accuracy_history:
        current_acc = accuracy_history[-1].get("accuracy", 0.94)
        prev_acc = accuracy_history[-2]["accuracy"] if len(accuracy_history) > 1 else current_acc
    else:
        current_acc = prev_acc = 0.94

    with col1:
        st.metric(
            "Accuracy (approx)",
            f"{current_acc:.1%}",
            delta=f"{current_acc - prev_acc:.1%}" if len(accuracy_history) > 1 else None,
        )

    sorted_lat = sorted(latencies) if latencies else [0]
    p50 = sorted_lat[len(sorted_lat) // 2]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)] if sorted_lat else 0
    with col2:
        st.metric("p50 Latency", f"{p50:.0f}ms")

    with col3:
        st.metric(
            "p99 Latency",
            f"{p99:.0f}ms",
            delta=f"SLA: {config['monitoring']['latency_p99_threshold_ms']}ms",
            delta_color="off",
        )

    success_rate = statuses.count("success") / len(statuses) if statuses else 0
    with col4:
        st.metric("Success Rate (window)", f"{success_rate:.1%}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Accuracy / success rate over time")
        import plotly.graph_objects as go

        fig = go.Figure()
        dates = [d["date"] for d in accuracy_history]
        accs = [d.get("accuracy", 0) for d in accuracy_history]

        fig.add_trace(go.Scatter(
            x=dates, y=accs,
            mode="lines+markers",
            name="Daily success ratio",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=4),
        ))

        threshold = config["monitoring"]["accuracy_threshold"]
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold:.0%})",
        )

        fig.update_layout(
            yaxis_title="Success ratio",
            yaxis_range=[0, 1.0],
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Latency Distribution")

        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=latencies,
            nbinsx=40,
            marker_color="#4CAF50",
            opacity=0.8,
            name="Latency (ms)",
        ))

        sla = config["monitoring"]["latency_p99_threshold_ms"]
        fig2.add_vline(
            x=sla,
            line_dash="dash",
            line_color="red",
            annotation_text=f"p99 SLA ({sla}ms)",
        )

        fig2.update_layout(
            xaxis_title="Latency (ms)",
            yaxis_title="Count",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    recent = _recent_extractions_cached()
    if recent:
        st.subheader("Recent extractions")
        st.dataframe(recent, use_container_width=True, hide_index=True)

    st.divider()

    # ── Drift monitoring panel ──────────────────────────────────────────────
    latest_report = _load_latest_report()
    if latest_report:
        _render_drift_panel(latest_report)
    else:
        with st.expander("Drift & SLA Status (no report yet)", expanded=False):
            st.info(
                "No monitoring report found. Run "
                "`python monitoring/monitor.py --full-report` "
                "to generate one."
            )

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Extraction status (window)")
        from collections import Counter
        status_counts = Counter(statuses)

        fig3 = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            marker_colors=["#4CAF50", "#FF9800", "#f44336", "#9E9E9E"],
            hole=0.4,
        )])
        fig3.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("Cache Performance")
        st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")
        st.metric("Memory Used", f"{cache_stats.get('used_memory_mb', 0):.0f} MB / 256 MB")
        st.metric("Total Hits", f"{cache_stats.get('hits', 0):,}")
        st.metric("Total Misses", f"{cache_stats.get('misses', 0):,}")

    with col_c:
        st.subheader("Alerts")
        alerts_path = Path("results/alerts.jsonl")
        if alerts_path.exists():
            alerts = []
            with open(alerts_path) as f:
                for line in f:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if alerts:
                for alert in alerts[-5:]:
                    severity = "🔴" if "CRITICAL" in alert["message"] else "🟡"
                    st.write(f"{severity} {alert['timestamp'][:16]}")
                    st.caption(alert["message"][:100])
            else:
                st.success("No alerts — system healthy ✓")
        else:
            st.success("No alerts — system healthy ✓")

    with st.expander("Configuration"):
        st.json({
            "model": config["model"]["base_model"],
            "accuracy_threshold": config["monitoring"]["accuracy_threshold"],
            "latency_sla_p99_ms": config["monitoring"]["latency_p99_threshold_ms"],
            "daily_sample_size": config["monitoring"]["daily_sample_size"],
            "cache_ttl": config["database"]["redis"]["cache_ttl_seconds"],
        })


if __name__ == "__main__":
    main()
