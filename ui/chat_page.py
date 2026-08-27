"""
Chat Page.
Main Q&A interface for the RAG assistant.
Displays chat history with inline citations and reliability panels.
"""

from __future__ import annotations

import streamlit as st

from core.pipeline import Pipeline, PipelineResult
from ui.components.reliability_panel import render_reliability_panel


def render_chat_page(
    pipeline: Pipeline,
    enable_rewrite: bool = True,
    enable_reranking: bool = False,
) -> None:
    """Render the main chat interface."""
    st.header("💬 Ask a Question")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show reliability panel for assistant messages
            if msg["role"] == "assistant" and "pipeline_result" in msg:
                result = msg["pipeline_result"]
                if result.reliability:
                    render_reliability_panel(result.reliability)

                # Show query rewrite info
                if result.processed_query and result.processed_query.was_rewritten:
                    _render_rewrite_info(result)

                # Show sources
                if result.retrieved_chunks:
                    _render_sources(result)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if documents are loaded
        if not pipeline.retriever.has_documents():
            no_docs_msg = (
                "📄 **No documents loaded yet.** Please upload documents using the sidebar first."
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": no_docs_msg,
                }
            )
            with st.chat_message("assistant"):
                st.warning(no_docs_msg)
            return

        # Run pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = pipeline.run(
                    query=prompt,
                    top_k=st.session_state.get("top_k", None),
                    temperature=st.session_state.get("temperature", 0.3),
                    enable_rewrite=enable_rewrite,
                    enable_reranking=enable_reranking,
                )

            # Display answer
            if result.should_abstain:
                st.warning(result.display_answer)
            else:
                st.markdown(result.display_answer)

            # Show reliability panel
            if result.reliability:
                render_reliability_panel(result.reliability)

            # Show query rewrite info
            if result.processed_query and result.processed_query.was_rewritten:
                _render_rewrite_info(result)

            # Show sources (color-coded by retrieval path)
            if result.retrieved_chunks:
                _render_sources(result)

            # Show latency + route info
            latency_parts = []
            if "query_processing" in result.latency_ms:
                latency_parts.append(f"rewrite: {result.latency_ms['query_processing']:.0f}ms")
            latency_parts.append(f"retrieval: {result.latency_ms.get('retrieval', 0):.0f}ms")
            latency_parts.append(f"generation: {result.latency_ms.get('generation', 0):.0f}ms")
            latency_parts.append(f"reliability: {result.latency_ms.get('reliability', 0):.0f}ms")

            route_label = f"route: {result.route_mode}"
            if result.reliability and result.reliability.sources_used:
                src = " + ".join(result.reliability.sources_used)
                route_label += f" ({src})"

            st.caption(
                f"⏱️ Total: {result.total_latency_ms:.0f}ms "
                f"({', '.join(latency_parts)}) · {route_label}"
            )

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.display_answer,
                "pipeline_result": result,
            }
        )


def _render_rewrite_info(result: PipelineResult) -> None:
    """Show query rewrite information."""
    pq = result.processed_query
    if not pq or not pq.was_rewritten:
        return

    with st.expander("🔄 Query was rewritten for better retrieval", expanded=False):
        st.markdown(f"**Original:** {pq.original}")
        st.markdown(f"**Rewritten:** {pq.rewritten}")


def _render_sources(result: PipelineResult) -> None:
    """Render retrieved sources as an expandable panel.

    Each source is color-coded by retrieval path:
    - vector chunks      → 🔵
    - graph traversal    → 🟣
    - community report   → 🟢
    """
    with st.expander(
        f"📚 Sources ({len(result.retrieved_chunks)} chunks retrieved)",
        expanded=False,
    ):
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            # Build a citation label that matches the generator's prefix scheme.
            prefix = {
                "vector": "V",
                "graph": "G",
                "community": "C",
            }.get(chunk.retrieval_source, "V")
            path_icon = {
                "vector": "🔵",
                "graph": "🟣",
                "community": "🟢",
            }.get(chunk.retrieval_source, "🔵")
            path_label = {
                "vector": "vector",
                "graph": "graph",
                "community": "community",
            }.get(chunk.retrieval_source, "vector")

            # Show both similarity and rerank scores if available
            score_label = f"Score: {chunk.score:.3f}"
            if chunk.rerank_score >= 0:
                score_label = f"Similarity: {chunk.score:.3f} → Rerank: {chunk.rerank_score:.3f}"
                effective = chunk.rerank_score
            else:
                effective = chunk.score

            score_color = "🟢" if effective >= 0.7 else "🟡" if effective >= 0.5 else "🔴"
            st.markdown(
                f"**[{prefix}{i}]** {path_icon} {path_label} · {score_color} {score_label} — "
                f"*{chunk.source}*"
            )
            st.text(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))
            st.divider()
