# Validation record

Validated on macOS, Python 3.12.13, 2026-09-08.

- Automated suite: **48 passed**, no skipped tests.
- Package statement coverage: **94.70%**; CI minimum is 85%.
- Ruff lint and formatting checks passed.
- Source distribution and wheel built; wheel installed in a fresh environment outside the repository.
- Installed CLI and packaged browser assets were exercised successfully.
- Chromium desktop and 390-pixel mobile workflows were inspected. No horizontal overflow was detected. Final browser console: zero errors and zero warnings.
- GitHub CI runs the test suite, packaging and installed-wheel smoke checks on Python 3.11, 3.12 and 3.13, plus a Docker build/execution job. Run results are available under the repository's Actions tab.

The browser check covered demo import, question submission, source display, graph selection, JSON answer download, file chooser upload, document deletion, and abstention for an unrelated question. Automated tests include a real generated PDF, HTML parsing, UTF-8 errors, chunk provenance, graph traversal, cache invalidation, corpus filtering, concurrent imports, and provider response contracts.

The bundled smoke evaluation retrieved the expected source in the top five for all five authored English/Chinese questions, and abstained on one unrelated question. This is a small deterministic fixture result, not a claim of general retrieval accuracy or model quality.

Live Ollama was not running and no paid model endpoint was called. Model HTTP protocols were tested with deterministic in-process transports. Live model task success, semantic embedding quality, cost and latency remain unmeasured. Docker was not running locally; the repository CI container job supplies that validation. Two upstream TestClient deprecation warnings occur in the Python test suite and do not affect the application browser console.
