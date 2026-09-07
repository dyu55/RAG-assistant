# 🔍 RAG Assistant — Reliable Knowledge Retrieval

A retrieval-augmented generation (RAG) assistant with **production-grade reliability engineering**: citation verification, grounding checks, confidence scoring, and automatic abstention on low-confidence answers.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Document Ingestion** | Upload PDF, Markdown, TXT, or HTML files into a vector knowledge base |
| 🔎 **Semantic Search** | ChromaDB-powered vector similarity search for relevant document chunks |
| 📝 **Constrained Generation** | LLM answers are forced to cite sources and admit uncertainty |
| ✅ **Citation Verification** | Checks whether the answer actually references retrieved chunks |
| 🔗 **Grounding Check** | Verifies that cited quotes exist in source documents (fuzzy matching) |
| 📊 **Confidence Scoring** | Weighted aggregate of retrieval, citation, grounding, and self-confidence signals |
| 🚫 **Automatic Abstention** | System refuses to answer when evidence is insufficient |
| 📈 **Full Logging** | Every query/response logged as structured JSONL for analysis |

## 🏗️ Architecture

```
User Question
    │
    ▼
┌─────────────────┐
│  Layer 1:       │
│  Retrieval      │ ─── ChromaDB vector search (top-k cosine similarity)
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Layer 2:       │
│  Generation     │ ─── Constrained JSON output with [Source N] citations
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Layer 3:       │
│  Reliability    │ ─── Citation check → Grounding → Confidence → Abstention
└────────┬────────┘
         │
    ▼
┌────────────┐
│  Answer    │ ─── Grounded answer with scores, OR abstention message
└────────────┘
```

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/dyu55/RAG-assistant.git
cd RAG-assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install .

# Optional local embedding backend
python -m pip install ".[local]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### 4. Use

1. Enter your OpenAI API key in the sidebar
2. Upload documents (PDF, MD, TXT, HTML)
3. Click "Ingest Documents"
4. Ask questions in the chat interface
5. Review the reliability report for each answer

## 📁 Project Structure

```
├── app.py                     # Streamlit entry point
├── config/settings.py         # Configuration (env vars, constants)
├── providers/openai_provider.py  # OpenAI API wrapper
├── ingestion/
│   ├── loader.py              # Document loading (PDF, MD, TXT, HTML)
│   ├── chunker.py             # Recursive text chunking with overlap
│   └── embedder.py            # Embedding generation + ChromaDB storage
├── core/
│   ├── models.py              # Shared retrieval and answer data contracts
│   ├── result.py              # Pipeline result and log serialization
│   ├── retrieval.py           # Routing and parallel retrieval
│   ├── cache.py               # Isolated LRU response cache
│   ├── retriever.py           # Vector similarity search
│   ├── generator.py           # Constrained answer generation
│   ├── reliability.py         # ⭐ Citation, grounding, confidence, abstention
│   └── pipeline.py            # Pipeline orchestrator
├── evaluation/
│   └── logger.py              # JSONL query logging
└── ui/
    ├── chat_page.py           # Chat interface
    └── components/
        └── reliability_panel.py  # Visual reliability scorecard
```

## ⭐ Reliability Engine (Core Differentiator)

The reliability engine (`core/reliability.py`) runs four independent checks on every generated answer:

### 1. Citation Presence Check
Verifies the answer includes citations referencing valid retrieved chunks.

### 2. Grounding Verification
Uses fuzzy string matching (`difflib.SequenceMatcher` + sliding window) to verify that cited quotes actually exist in the source documents. Catches fabricated citations.

### 3. Confidence Scoring
Weighted aggregate of four signals:
- **Retrieval quality** (30%): Average similarity score of top-k chunks
- **Citation coverage** (25%): Are sources properly cited?
- **Grounding score** (25%): Are citations verified?
- **Self-confidence** (20%): Model's own uncertainty estimate

### 4. Abstention Logic
System abstains when:
- Overall confidence falls below threshold (default: 0.6)
- Best retrieval score is too low (no relevant documents)
- Zero citations AND zero grounding (no evidence trail)

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o-mini (configurable)
- **Embeddings**: OpenAI text-embedding-3-small or local sentence-transformers
- **Vector Store**: ChromaDB (persistent, local)
- **UI**: Streamlit
- **Document Processing**: PyMuPDF (PDF), built-in (MD, TXT, HTML)

## 📊 Logging & Evaluation

Every query is logged to `data/logs/queries.jsonl` with:
- Timestamp, query, answer, citations
- Full reliability report (scores, verdict, abstention reason)
- Per-layer latency (retrieval, generation, reliability check)
- Model name and configuration

## Development and cache lifecycle

```bash
python -m pip install -e ".[dev]"
python -m pytest
make lint
make security
python -m build
```

RRF controls ranking through `chunk.metadata["rrf_score"]`; `chunk.score` remains the original relevance signal used by CRAG and reliability checks. Cache hits retain the full evidence and reliability report and are logged like fresh answers.

Applications that inject `SemanticCache` into a long-lived `Pipeline` must call `pipeline.invalidate_cache()` after changing documents or pipeline configuration. Cache namespaces isolate pipeline instances and generation/retrieval options. The built-in pipeline uses exact response caching; semantic lookup is available through `SemanticCache.get(..., query_embedding=...)` when a caller supplies an embedding.

The default Docker image uses API embeddings. To include local embedding models, build with `docker build --build-arg INSTALL_LOCAL_EMBEDDINGS=true -t rag-assistant:local .`.

See [the refactoring report](docs/REFACTORING.md) for validation and compatibility details.

## License

MIT
