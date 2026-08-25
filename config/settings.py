"""
Global settings for the RAG Assistant.
Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# ── LLM Provider ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding model and backend
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")
# Optional: custom OpenAI-compatible embedding endpoint (e.g., vLLM, Ollama)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")

# ── Embedding Backend ──────────────────────────────────────────────────────────
# "openai" uses OpenAI API embeddings
# "local" uses sentence-transformers/all-MiniLM-L6-v2 (free, no API key needed)
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))

# ── Reliability ───────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Weights for confidence scoring (must sum to 1.0)
CONFIDENCE_WEIGHTS = {
    "retrieval": 0.30,
    "citation": 0.25,
    "grounding": 0.25,
    "self_confidence": 0.20,
}

# Minimum retrieval score to consider a chunk relevant
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.3"))

# Grounding: minimum fuzzy match ratio to consider a quote verified
GROUNDING_MATCH_THRESHOLD = float(os.getenv("GROUNDING_MATCH_THRESHOLD", "0.6"))

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = str(PROJECT_ROOT / os.getenv("CHROMA_DB_PATH", "data/chroma_db"))
LOG_DIR = str(PROJECT_ROOT / os.getenv("LOG_DIR", "data/logs"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ── Supported File Types ──────────────────────────────────────────────────────
SUPPORTED_FILE_TYPES = {".pdf", ".md", ".txt", ".html"}

# ── GraphRAG / Neo4j ───────────────────────────────────────────────────────────
USE_GRAPH_RAG = os.getenv("USE_GRAPH_RAG", "true").lower() == "true"
GRAPH_RAG_MODE = os.getenv("GRAPH_RAG_MODE", "auto")  # off | auto | local | global | both
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

COMMUNITY_LEVELS = int(os.getenv("COMMUNITY_LEVELS", "3"))
GRAPH_LOCAL_TOPK = int(os.getenv("GRAPH_LOCAL_TOPK", "5"))
GRAPH_LOCAL_HOPS = int(os.getenv("GRAPH_LOCAL_HOPS", "2"))
GRAPH_GLOBAL_TOP_COMMUNITIES = int(os.getenv("GRAPH_GLOBAL_TOP_COMMUNITIES", "5"))
GRAPH_REBUILD_THRESHOLD = int(os.getenv("GRAPH_REBUILD_THRESHOLD", "100"))

# Citation prefixes used by the generator. [V] = vector, [G] = graph traversal,
# [C] = community report. The plan proposes this format.
CITATION_PREFIX_VECTOR = "V"
CITATION_PREFIX_GRAPH = "G"
CITATION_PREFIX_COMMUNITY = "C"
