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

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Embedding Backend ─────────────────────────────────────────────────────────
# "openai" uses OpenAI API embeddings
# "local" uses sentence-transformers/all-MiniLM-L6-v2 (free, no API key needed)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")
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
