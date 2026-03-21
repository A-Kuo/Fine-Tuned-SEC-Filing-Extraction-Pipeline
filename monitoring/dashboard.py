"""Streamlit Monitoring Dashboard.

Visualizes model performance, latency distributions, accuracy drift,
and system health in a single-page dashboard. Designed for ops teams
to quickly assess whether the extraction system is healthy.

Usage:
    streamlit run monitoring/dashboard.py
    # Opens on http://localhost:8501
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config


def generate_demo_data():
    """Generate realistic demo data for the dashboard.

    In production, this data comes from PostgreSQL via DatabaseManager.
    For the prototype, we generate plausible time series.
    """
    random.seed(42)
    now = datetime.utcnow()

    # Daily accuracy over 30 days (stable ~94%, small variance)
    accuracy_history = []
    for i in range(30):
        day = now - timedelta(days=29 - i)
        acc = 0.94 + random.gauss(0, 0.008)
        accuracy_history.append({
            "date": day.strftime("%Y-%m-%d"),
            "accuracy": max(0.88, min(0.98, acc)),
            "sample_size": random.randint(40, 60),
        })

    # Latency samples (last 24 hours)
    latencies = [max(100, random.gauss(350, 80)) for _ in range(500)]

    # Extraction log (last 100)
    statuses = random.choices(
        ["success", "validation_error", "parse_error", "timeout"],
        weights=[94, 3, 2, 1],
        k=100,
    )

    # Cache stats
    cache_stats = {
        "hits": 9823,
        "misses": 177,
        "hit_rate": 0.982,
        "used_memory_mb": 48.3,
    }

    return accuracy_history, latencies, statuses, cache_stats


def main():
    st.set_page_config(
        page_title="Financial LLM Monitor",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Financial LLM Extraction Monitor")
    st.caption("Real-time monitoring for SEC filing extraction system")

    config = load_config()
    accuracy_history, latencies, statuses, cache_stats = generate_demo_data()

    # ── Top-level KPIs ──
    col1, col2, col3, col4 = st.columns(4)

    current_acc = accuracy_history[-1]["accuracy"]
    with col1:
        st.metric(
            "Accuracy",
            f"{current_acc:.1%}",
            delta=f"{current_acc - accuracy_history[-2]['accuracy']:.1%}",
        )

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
    with col2:
        st.metric("p50 Latency", f"{p50:.0f}ms")

    with col3:
        st.metric(
            "p99 Latency",
            f"{p99:.0f}ms",
            delta=f"SLA: {config['monitoring']['latency_p99_threshold_ms']}ms",
            delta_color="off",
        )

    success_rate = statuses.count("success") / len(statuses)
    with col4:
        st.metric("Success Rate (24h)", f"{success_rate:.1%}")

    st.divider()

    # ── Accuracy Trend ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Accuracy Over Time")
        import plotly.graph_objects as go

        fig = go.Figure()
        dates = [d["date"] for d in accuracy_history]
        accs = [d["accuracy"] for d in accuracy_history]

        fig.add_trace(go.Scatter(
            x=dates, y=accs,
            mode="lines+markers",
            name="Daily Accuracy",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=4),
        ))

        # Threshold line
        threshold = config["monitoring"]["accuracy_threshold"]
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({threshold:.0%})",
        )

        fig.update_layout(
            yaxis_title="Accuracy",
            yaxis_range=[0.85, 1.0],
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

        # SLA line
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

    st.divider()

    # ── System Health ──
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Extraction Status (24h)")
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
        st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1%}")
        st.metric("Memory Used", f"{cache_stats['used_memory_mb']:.0f} MB / 256 MB")
        st.metric("Total Hits", f"{cache_stats['hits']:,}")
        st.metric("Total Misses", f"{cache_stats['misses']:,}")

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
                for alert in alerts[-5:]:  # Last 5 alerts
                    severity = "🔴" if "CRITICAL" in alert["message"] else "🟡"
                    st.write(f"{severity} {alert['timestamp'][:16]}")
                    st.caption(alert["message"][:100])
            else:
                st.success("No alerts — system healthy ✓")
        else:
            st.success("No alerts — system healthy ✓")

    # ── Configuration ──
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
