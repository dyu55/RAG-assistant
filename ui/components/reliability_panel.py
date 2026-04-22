"""
Reliability Panel Component.
Renders the visual reliability scorecard in Streamlit.
Shows citation score, grounding score, confidence, unsupported claims, and verdict.
"""
from __future__ import annotations

import streamlit as st
from core.reliability import ReliabilityReport


def render_reliability_panel(report: ReliabilityReport) -> None:
    """Render the reliability scorecard as an expandable Streamlit panel."""
    verdict_labels = {
        "grounded": ("✅ Grounded", "success"),
        "partially_grounded": ("🟡 Partially Grounded", "warning"),
        "low_confidence": ("🟠 Low Confidence", "warning"),
        "abstained": ("🚫 Abstained", "error"),
    }

    label, alert_type = verdict_labels.get(
        report.verdict, ("❓ Unknown", "info")
    )

    with st.expander(f"📊 Reliability Report — {label}", expanded=False):
        # ── Score Cards (2 rows × 2 columns for readability) ──────────
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            _score_card("🎯 Confidence", report.confidence)
        with row1_col2:
            faithfulness = 1.0 - report.unsupported_ratio
            _score_card("📋 Faithfulness", faithfulness)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            _score_card("📝 Citation", report.citation_score)
        with row2_col2:
            _score_card("🔗 Grounding", report.grounding_score)

        st.markdown("")  # spacer

        # ── Verdict ───────────────────────────────────────────────────
        if report.should_abstain:
            st.error(f"**Abstention Reason:** {report.abstention_reason}")
        elif report.confidence >= 0.8:
            st.success("Answer is well-grounded in the retrieved documents.")
        elif report.confidence >= 0.6:
            st.warning("Answer has partial support. Some claims may need verification.")
        else:
            st.warning("Answer has weak support. Treat with caution.")

        # ── Grounding Details ─────────────────────────────────────────
        if report.grounding_details:
            st.markdown("**Citation Grounding Details:**")
            for i, detail in enumerate(report.grounding_details, 1):
                status_icon = {
                    "strong": "🟢",
                    "partial": "🟡",
                    "ungrounded": "🔴",
                }.get(detail.status, "❓")

                st.markdown(
                    f"{status_icon} **Citation {i}** — "
                    f"Match: {detail.match_ratio:.0%} ({detail.status})"
                )
                if detail.citation.quote:
                    st.caption(f'Quote: "{detail.citation.quote[:150]}..."')

        # ── Unsupported Claims ────────────────────────────────────────
        unsupported = [c for c in report.unsupported_claims if not c.is_supported]
        if unsupported:
            st.markdown(f"**⚠️ Unsupported Claims ({len(unsupported)}):**")
            for claim in unsupported:
                st.markdown(
                    f"🔴 \"{claim.claim[:120]}{'...' if len(claim.claim) > 120 else ''}\" "
                    f"— best match: {claim.best_match_score:.0%}"
                )
        elif report.unsupported_claims:
            st.markdown(f"✅ All {len(report.unsupported_claims)} claims are supported by source documents.")

        # ── Raw Details ───────────────────────────────────────────────
        with st.expander("🔍 Raw Details"):
            st.json(report.details)


def _score_card(label: str, score: float) -> None:
    """Render a score as a clearly visible metric card."""
    if score >= 0.8:
        color = "🟢"
    elif score >= 0.6:
        color = "🟡"
    elif score >= 0.3:
        color = "🟠"
    else:
        color = "🔴"

    st.metric(
        label=f"{label}",
        value=f"{color} {score:.0%}",
    )
