"""
RAG Assistant — Streamlit App
A reliable retrieval-augmented knowledge assistant with citation verification,
grounding checks, confidence scoring, and abstention logic.
"""
from __future__ import annotations

import streamlit as st
import logging
import uuid

from config import settings
from providers.custom_provider import CustomProvider
from ingestion.loader import load_from_bytes
from ingestion.chunker import RecursiveChunker
from ingestion.embedder import Embedder
from core.retriever import Retriever
from core.generator import Generator
from core.reliability import ReliabilityChecker
from core.query_handler import QueryHandler
from core.pipeline import Pipeline
from evaluation.logger import QueryLogger
from ui.chat_page import render_chat_page
from ui.dashboard_page import render_dashboard_page

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Prevent metric value truncation */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        overflow: visible !important;
        white-space: nowrap !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        overflow: visible !important;
        white-space: nowrap !important;
    }
    /* Metric card styling */
    [data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        padding: 12px 8px;
        border-radius: 8px;
        text-align: center;
        min-width: 0;
    }
    /* Subtle dividers */
    hr {
        border: none;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }
    /* Make expander content not clip */
    .streamlit-expanderContent {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ─────────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "provider_type": "openai",
        "api_key": "",
        "base_url": "",
        "pipeline_initialized": False,
        "embedding_backend": settings.EMBEDDING_BACKEND,
        "model": settings.OPENAI_MODEL,
        "temperature": 0.3,
        "top_k": settings.TOP_K,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "docs_ingested": 0,
        "enable_rewrite": True,
        "enable_reranking": False,
        "active_page": "Chat",
        "collection_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with settings and document upload."""
    with st.sidebar:
        st.title("🔍 RAG Assistant")
        st.caption("Reliable Knowledge Retrieval with Citation Verification")

        st.divider()

        # ── Provider Setup ─────────────────────────────────────────────
        st.subheader("🔑 LLM Provider")
        st.session_state.provider_type = st.selectbox(
            "Provider",
            ["openai", "deepseek", "groq", "together", "anthropic", "custom"],
            index=0,
            help="Select your LLM provider. 'custom' lets you set any OpenAI-compatible endpoint.",
        )

        api_key = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            help=f"API key for {st.session_state.provider_type}. Not stored permanently.",
        )

        # Base URL — only for "custom" provider
        if st.session_state.provider_type == "custom":
            base_url = st.text_input(
                "Base URL",
                value=st.session_state.base_url,
                placeholder="https://api.example.com/v1",
                help="OpenAI-compatible base URL (e.g., your vLLM or Ollama server).",
            )
        else:
            base_url = st.session_state.base_url

        # ── Model ──────────────────────────────────────────────────────
        from providers.custom_provider import PROVIDER_MODELS
        all_providers = ["openai", "deepseek", "groq", "together", "anthropic", "custom"]
        models_for_provider = PROVIDER_MODELS.get(st.session_state.provider_type, [])
        if models_for_provider:
            model_options = models_for_provider
            model_index = 0
            if st.session_state.model in model_options:
                model_index = model_options.index(st.session_state.model)
            st.session_state.model = st.selectbox(
                "Model",
                model_options,
                index=model_index,
                help=f"Model for {st.session_state.provider_type}.",
            )
        else:
            st.session_state.model = st.text_input(
                "Model",
                value=st.session_state.model,
                placeholder="meta-llama/Llama-3-70b",
                help="Enter the model name for your custom endpoint.",
            )

        # Track changes and reset pipeline
        changed = (
            api_key != st.session_state.api_key
            or st.session_state.provider_type != st.session_state.get("_prev_provider_type")
            or st.session_state.model != st.session_state.get("_prev_model")
            or base_url != st.session_state.base_url
        )
        st.session_state.api_key = api_key
        st.session_state.base_url = base_url
        st.session_state["_prev_provider_type"] = st.session_state.provider_type
        st.session_state["_prev_model"] = st.session_state.model
        if changed:
            st.session_state.pipeline_initialized = False

        st.divider()

        # ── Model Settings ───────────────────────────────────────────
        st.subheader("⚙️ Settings")

        st.session_state.embedding_backend = st.selectbox(
            "Embedding Backend",
            ["openai", "local"],
            index=0 if st.session_state.embedding_backend == "openai" else 1,
            help="'openai' uses text-embedding-3-small. 'local' uses all-MiniLM-L6-v2 (free).",
        )

        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.3, 0.1,
            help="Lower = more deterministic answers.",
        )

        st.session_state.top_k = st.slider(
            "Top-K Chunks", 1, 15, settings.TOP_K,
            help="Number of document chunks to retrieve.",
        )

        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, settings.CONFIDENCE_THRESHOLD, 0.05,
            help="Below this threshold, the system will abstain from answering.",
        )

        st.divider()

        # ── Phase 2: Reliability Enhancements ─────────────────────────
        st.subheader("🔬 Reliability")

        st.session_state.enable_rewrite = st.toggle(
            "Query Rewrite",
            value=st.session_state.enable_rewrite,
            help="Rewrites vague queries into more specific search queries before retrieval.",
        )

        st.session_state.enable_reranking = st.toggle(
            "LLM Reranking",
            value=st.session_state.enable_reranking,
            help="Re-scores retrieved chunks using the LLM for better precision. Adds latency.",
        )

        st.divider()

        # ── Document Upload ──────────────────────────────────────────
        st.subheader("📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload files to the knowledge base",
            type=["pdf", "md", "txt", "html"],
            accept_multiple_files=True,
            help="Supported: PDF, Markdown, TXT, HTML",
        )

        if uploaded_files and st.button("📥 Ingest Documents", type="primary"):
            _ingest_documents(uploaded_files)

        # ── Collection Stats ─────────────────────────────────────────
        st.divider()
        st.subheader("📊 Knowledge Base")
        _render_collection_stats()

        # ── Log Stats ────────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Query Stats")
        _render_log_stats()


def _ingest_documents(uploaded_files):
    """Process and ingest uploaded documents."""
    provider_type = st.session_state.provider_type
    api_key = st.session_state.api_key
    model = st.session_state.model
    base_url = st.session_state.base_url

    if not api_key:
        st.error("Please enter your API key first.")
        return

    try:
        # Use CustomProvider for both LLM and embedding
        provider = CustomProvider(
            provider=provider_type,
            api_key=api_key,
            model=model,
            base_url=base_url if provider_type == "custom" else None,
        )
        embedder = Embedder(backend="openai", openai_provider=provider)

        chunker = RecursiveChunker()

        progress = st.progress(0, text="Processing documents...")
        total_chunks = 0

        for i, file in enumerate(uploaded_files):
            progress.progress(
                (i) / len(uploaded_files),
                text=f"Processing {file.name}...",
            )

            # Determine file type
            suffix = "." + file.name.rsplit(".", 1)[-1].lower()

            # Load document
            doc = load_from_bytes(
                file_bytes=file.read(),
                filename=file.name,
                file_type=suffix,
            )

            # Chunk document
            doc_id = str(uuid.uuid4())
            chunks = chunker.chunk_document(
                text=doc.content,
                doc_id=doc_id,
                metadata=doc.metadata,
            )

            # Ingest into ChromaDB
            count = embedder.ingest(chunks)
            total_chunks += count

        progress.progress(1.0, text="Done!")
        st.success(
            f"✅ Ingested {len(uploaded_files)} file(s), "
            f"{total_chunks} chunks into the knowledge base."
        )
        st.session_state.docs_ingested += len(uploaded_files)
        st.session_state.pipeline_initialized = False  # Force re-init

        # Refresh cached collection stats
        _refresh_collection_stats()

    except Exception as e:
        st.error(f"❌ Ingestion failed: {str(e)}")


def _render_collection_stats():
    """Show ChromaDB collection statistics.

    Uses cached stats from session state (refreshed after ingestion).
    Falls back to direct ChromaDB query if cache is empty (e.g., on first load).
    """
    try:
        # Use cached stats if available, otherwise fetch directly
        stats = st.session_state.get("collection_stats")
        if stats is None:
            stats = _get_embedder_stats()

        if stats and stats["total_chunks"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats["total_documents"])
            with col2:
                st.metric("Chunks", stats["total_chunks"])

            if stats["filenames"]:
                with st.expander("📁 Files"):
                    for f in stats["filenames"]:
                        st.caption(f"• {f}")
        else:
            st.caption("No documents loaded yet.")
    except Exception:
        st.caption("No documents loaded yet.")


def _render_log_stats():
    """Show aggregate query statistics from logs."""
    try:
        query_logger = QueryLogger()
        stats = query_logger.get_log_stats()
        if stats["total_queries"] > 0:
            st.caption(
                f"📊 **{stats['total_queries']}** queries | "
                f"Confidence: **{stats['avg_confidence']:.0%}** | "
                f"Abstain: **{stats['abstention_rate']:.0%}**"
            )
            st.caption(f"⏱️ Avg latency: **{stats['avg_latency_ms']:.0f}ms**")
        else:
            st.caption("No queries logged yet.")
    except Exception:
        st.caption("No queries logged yet.")


def _get_embedder_stats():
    """Get collection stats directly from ChromaDB without full embedder init.

    Used as fallback when no API key is available (pre-ingestion state).
    Prefer _refresh_collection_stats() + cached st.session_state.collection_stats
    for production use after pipeline is initialized.
    """
    try:
        import chromadb
        from pathlib import Path

        db_path = Path(settings.CHROMA_DB_PATH)
        if not db_path.exists():
            return None

        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()

        if count > 0:
            result = collection.get(limit=min(count, 10000), include=["metadatas"])
            doc_ids = set()
            filenames = set()
            for meta in result["metadatas"]:
                doc_ids.add(meta.get("doc_id", "unknown"))
                filenames.add(meta.get("filename", "unknown"))
        else:
            doc_ids = set()
            filenames = set()

        return {
            "collection_name": collection.name,
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "filenames": sorted(filenames),
        }
    except Exception:
        return None


def _refresh_collection_stats():
    """Refresh and cache collection stats in session state.

    Called after document ingestion to update the cached stats.
    Avoids re-scanning the full collection on every sidebar render.
    """
    try:
        import chromadb
        from pathlib import Path

        db_path = Path(settings.CHROMA_DB_PATH)
        if not db_path.exists():
            st.session_state.collection_stats = None
            return

        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()

        if count > 0:
            result = collection.get(limit=min(count, 10000), include=["metadatas"])
            doc_ids = set()
            filenames = set()
            for meta in result["metadatas"]:
                doc_ids.add(meta.get("doc_id", "unknown"))
                filenames.add(meta.get("filename", "unknown"))
        else:
            doc_ids = set()
            filenames = set()

        st.session_state.collection_stats = {
            "collection_name": collection.name,
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "filenames": sorted(filenames),
        }
    except Exception:
        st.session_state.collection_stats = None


def _get_pipeline():
    """Initialize or retrieve the pipeline from session state."""
    provider_type = st.session_state.provider_type
    api_key = st.session_state.api_key
    model = st.session_state.model
    base_url = st.session_state.base_url

    if not api_key:
        return None

    # Cache key based on settings
    cache_key = f"{provider_type}_{api_key[:8] if api_key else 'local'}_{model}"

    if (
        st.session_state.pipeline_initialized
        and st.session_state.get("pipeline_cache_key") == cache_key
        and "pipeline" in st.session_state
    ):
        return st.session_state.pipeline

    try:
        # Always use CustomProvider for the LLM
        provider = CustomProvider(
            provider=provider_type,
            api_key=api_key,
            model=model,
            base_url=base_url if provider_type == "custom" else None,
        )
        embedder = Embedder(backend="openai", openai_provider=provider)

        retriever = Retriever(
            embedder=embedder,
            rerank_provider=provider,  # LLM reranking uses the same provider
        )
        generator = Generator(provider=provider)

        # Update confidence threshold
        settings.CONFIDENCE_THRESHOLD = st.session_state.confidence_threshold

        reliability_checker = ReliabilityChecker()
        query_handler = QueryHandler(provider=provider)
        query_logger = QueryLogger()

        pipeline = Pipeline(
            retriever=retriever,
            generator=generator,
            reliability_checker=reliability_checker,
            query_handler=query_handler,
            query_logger=query_logger,
        )

        st.session_state.pipeline = pipeline
        st.session_state.pipeline_initialized = True
        st.session_state.pipeline_cache_key = cache_key

        return pipeline

    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Page navigation
    page = st.session_state.get("active_page", "Chat")
    selected = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Dashboard"],
        index=0 if page == "Chat" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.active_page = "Chat" if "Chat" in selected else "Dashboard"

    if st.session_state.active_page == "Dashboard":
        render_dashboard_page()
        return

    # Chat page
    pipeline = _get_pipeline()

    if not st.session_state.api_key:
        st.info(
            "👋 Welcome to **RAG Assistant**! \n\n"
            "Enter your OpenAI API key in the sidebar to get started. "
            "Then upload documents and start asking questions.\n\n"
            "**Features:**\n"
            "- 📄 Document ingestion (PDF, Markdown, TXT, HTML)\n"
            "- 🔍 Semantic search with ChromaDB\n"
            "- 📝 Citation-aware answer generation\n"
            "- ✅ Reliability scoring (citation, grounding, confidence)\n"
            "- 🚫 Automatic abstention on low-confidence answers\n"
            "- 🔬 Query rewrite & LLM reranking\n"
            "- 📊 Evaluation dashboard"
        )
        return

    if pipeline is None:
        st.error("Could not initialize the pipeline. Check your API key and settings.")
        return

    render_chat_page(
        pipeline,
        enable_rewrite=st.session_state.enable_rewrite,
        enable_reranking=st.session_state.enable_reranking,
    )


if __name__ == "__main__":
    main()
