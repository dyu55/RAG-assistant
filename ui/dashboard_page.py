"""
Dashboard Page.
Evaluation dashboard showing aggregate metrics, query history, and failure analysis.
"""

from __future__ import annotations

import streamlit as st

from evaluation.logger import QueryLogger
from evaluation.metrics import compute_aggregate_metrics


def render_dashboard_page() -> None:
    """Render the evaluation dashboard."""
    st.header("📊 Evaluation Dashboard")

    query_logger = QueryLogger()
    entries = query_logger.get_recent_logs(n=500)

    if not entries:
        st.info("No query data yet. Ask some questions first to see metrics here.")
        return

    # ── Aggregate Metrics (2 rows × 3 columns) ───────────────────────────
    agg = compute_aggregate_metrics(entries)

    st.subheader("📈 Aggregate Metrics")

    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        st.metric("Total Queries", agg.total_queries)
    with row1_c2:
        st.metric("Avg Confidence", f"{agg.avg_confidence:.0%}")
    with row1_c3:
        st.metric("Avg Latency", f"{agg.avg_latency_ms:.0f}ms")

    row2_c1, row2_c2, row2_c3 = st.columns(3)
    with row2_c1:
        st.metric("Avg Faithfulness", f"{agg.avg_faithfulness:.0%}")
    with row2_c2:
        st.metric("Avg Grounding", f"{agg.avg_grounding_rate:.0%}")
    with row2_c3:
        st.metric("Abstention Rate", f"{agg.abstention_rate:.0%}")

    st.divider()

    # ── Confidence Distribution ───────────────────────────────────────────
    st.subheader("🎯 Confidence Distribution")
    confidences = []
    for entry in entries:
        rel = entry.get("reliability", {}) or {}
        conf = rel.get("confidence")
        if conf is not None:
            confidences.append(conf)

    if confidences:
        bins = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
        for c in confidences:
            if c < 0.2:
                bins["0-20%"] += 1
            elif c < 0.4:
                bins["20-40%"] += 1
            elif c < 0.6:
                bins["40-60%"] += 1
            elif c < 0.8:
                bins["60-80%"] += 1
            else:
                bins["80-100%"] += 1

        st.bar_chart(bins)

    st.divider()

    # ── Per-Layer Latency ─────────────────────────────────────────────────
    st.subheader("⏱️ Average Latency by Layer")
    layer_totals = {}
    layer_counts = {}
    for entry in entries:
        latency = entry.get("latency_ms", {})
        for layer, ms in latency.items():
            layer_totals[layer] = layer_totals.get(layer, 0) + ms
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    if layer_totals:
        avg_latencies = {
            layer: round(layer_totals[layer] / layer_counts[layer], 1) for layer in layer_totals
        }
        st.bar_chart(avg_latencies)

    st.divider()

    # ── Recent Queries Table ──────────────────────────────────────────────
    st.subheader("📋 Recent Queries")

    table_data = []
    for entry in reversed(entries[-20:]):
        rel = entry.get("reliability", {}) or {}
        verdict = rel.get("verdict", "unknown")

        verdict_emoji = {
            "grounded": "✅",
            "partially_grounded": "🟡",
            "low_confidence": "🟠",
            "abstained": "🚫",
        }.get(verdict, "❓")

        table_data.append(
            {
                "Time": entry.get("timestamp", "")[:19],
                "Query": entry.get("query", "")[:50],
                "Verdict": f"{verdict_emoji} {verdict}",
                "Conf.": f"{rel.get('confidence', 0):.0%}",
                "Ground.": f"{rel.get('grounding_score', 0):.0%}",
                "Latency": f"{entry.get('total_latency_ms', 0):.0f}ms",
            }
        )

    if table_data:
        st.dataframe(table_data, use_container_width=True)

    st.divider()

    # ── Failure Analysis ──────────────────────────────────────────────────
    st.subheader("⚠️ Failure Analysis")

    abstained_entries = [e for e in entries if e.get("should_abstain")]
    low_confidence = [
        e
        for e in entries
        if (e.get("reliability", {}) or {}).get("confidence", 1.0) < 0.6
        and not e.get("should_abstain")
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Abstained", len(abstained_entries))
        if abstained_entries:
            with st.expander("View abstained queries"):
                for entry in abstained_entries[-5:]:
                    reason = (entry.get("reliability", {}) or {}).get(
                        "abstention_reason", "Unknown"
                    )
                    st.markdown(f"**Q:** {entry.get('query', '')[:80]}")
                    st.caption(f"Reason: {reason}")
                    st.divider()

    with col2:
        st.metric("Low Confidence", len(low_confidence))
        if low_confidence:
            with st.expander("View low-confidence queries"):
                for entry in low_confidence[-5:]:
                    conf = (entry.get("reliability", {}) or {}).get("confidence", 0)
                    st.markdown(f"**Q:** {entry.get('query', '')[:80]}")
                    st.caption(f"Confidence: {conf:.0%}")
                    st.divider()
