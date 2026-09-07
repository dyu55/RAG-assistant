# Core refactoring — 2026-09-06

## Changes

- Moved shared retrieval/answer dataclasses into `core/models.py` and pipeline output into `core/result.py`.
- Extracted routing, concurrent retrieval and fusion into `core/retrieval.py`. The pipeline now coordinates explicit processing, correction, generation, verification and logging stages.
- Preserved relevance scores during RRF. Rank scores live in `metadata["rrf_score"]`, so a highly relevant document is no longer treated as low-confidence merely because its RRF score is about 0.016.
- Replaced the cache's duplicate list/map bookkeeping with one locked LRU store. Replacements, expiry, payload copying and namespaces have explicit behavior.
- Cache hits preserve retrieved sources, citations, generated answers and reliability reports. Hits are logged. Cache failures fall back to normal processing; generation/reliability failures abstain and are not cached.
- Fixed router failure and unavailable-graph fallback to vector retrieval. Backend timings measure work inside each worker.
- Defined package discovery and dependencies. Local embedding models are an optional installation extra. Removed generated coverage and trace files from version control while keeping the local files.

## Compatibility and boundaries

Existing public imports from `core.pipeline`, `core.generator` and `core.retriever` remain available. Consumers using the fusion score must use `metadata["rrf_score"]`; `score` now correctly remains a relevance signal.

For a long-lived pipeline with an injected cache, call `invalidate_cache()` after document or configuration changes. Pipeline caching uses exact query matching; semantic cache lookups require an explicitly supplied query embedding. Independent pipeline instances and run options cannot reuse each other's cached answers.

Backend clients own network deadlines. A Python future timeout cannot cancel an in-flight backend operation.

## Local validation

Environment: macOS arm64, Python 3.12.13.

| Check | Observed result |
| --- | --- |
| Original test baseline | 253 passed |
| Refactored full suite | 276 passed, 8.51 seconds |
| Coverage of core/graph/ingestion/evaluation/providers | 69% |
| Ruff lint and formatting | Passed |
| Existing Bandit gate | Passed |
| Source distribution and wheel | Built successfully |
| Wheel installation outside checkout | Imports passed |
| Installed Streamlit page via AppTest | Executed without exceptions |

The Streamlit check also exercised graceful startup with Neo4j unavailable. It did not validate a live Neo4j database, paid model responses, local embedding model downloads, retrieval accuracy on a benchmark corpus, or Docker execution. Third-party PyMuPDF deprecation warnings remain.

Reproduce with `python -m pip install -e ".[dev]"`, `python -m pytest`, `make lint`, `make security`, and `python -m build`.
