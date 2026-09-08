.PHONY: install test lint check run demo build
install:
	uv sync --extra dev
test:
	uv run pytest --cov=rag_assistant --cov-report=term-missing --cov-fail-under=85
lint:
	uv run ruff check .
	uv run ruff format --check .
check: lint test
run:
	uv run rag-assistant serve
demo:
	uv run rag-assistant demo
build:
	uv run python -m build
