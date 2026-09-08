# Design and migration

The package has one application service, shared by HTTP and CLI entry points. Parsing, provider I/O, retrieval, evidence checking, and storage have separate contracts. Browser assets ship in the wheel; a Node toolchain is not required.

## Index consistency

Each filename is a document identity. A content digest makes repeated imports idempotent. All parsing and embedding completes before a replacement transaction begins. The SQLite transaction deletes the previous document and cascades its chunks, inserts the new document/chunks, validates the configured index size and embedding identity, then increments a corpus revision. Queries read revision and chunks from one snapshot. Delete increments the same revision. Cache keys include revision and query/filter/mode settings, so old evidence cannot be reused for a new corpus.

The current implementation scans in-memory chunk snapshots for BM25 and vector similarity. It is intended for personal collections, with a default 5,000-chunk cap. This is an explicit scale boundary, not a claim of distributed search capacity.

## Retrieval and evidence

The three retrieval channels run concurrently. Reciprocal rank fusion operates on ranks, while relevance remains a separate field. A graph edge records two extracted names in the same passage. Bounded breadth-first traversal explores at most two hops and keeps a source-linked entity path. Names are extracted using explicit `[[entity]]` markers and capitalized terms; this is a lightweight association graph, not an LLM-extracted factual knowledge graph.

Model generation receives untrusted evidence as data and returns a validated claim array. Every claim must reference a retrieved source and provide a quotation found in that source. A lexical-overlap check catches obvious unsupported paraphrases, but cannot prove entailment, detect every negation/numeric contradiction, or replace human source review. The offline path extracts actual sentences without invoking a model.

## Migrating from 0.2

The new package uses `src/rag_assistant/`, `rag-assistant serve`, and `.rag-assistant/knowledge.sqlite3`. The earlier entry point, ChromaDB collections, Neo4j graph, API configuration names and Python imports are not compatible with this new implementation.

Keep the earlier database directories as archives and import the original PDF/Markdown/TXT/HTML documents into a new data directory. Do not point the new application at an existing ChromaDB database. Credentials remain external environment variables and are never inferred from old configuration.

The prior source is retained under Git tag `before-greenfield-v1-20260908`. To inspect it without changing the new branch, create a separate worktree at that tag. Reverting the rewrite commit restores the earlier tracked source.
