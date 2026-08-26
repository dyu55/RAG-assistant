.PHONY: help install dev-install test lint format security ci docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make dev-install  - Install production + dev/CI dependencies"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make lint         - Run ruff lint check"
	@echo "  make format       - Auto-format code with ruff"
	@echo "  make security     - Run bandit security scan"
	@echo "  make ci           - Run full local CI pre-flight pipeline"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Run app and Neo4j via docker-compose"
	@echo "  make clean        - Clean temporary caches and test artifacts"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

security:
	bandit -r core/ graph/ ingestion/ evaluation/ providers/ -ll -q

ci:
	bash scripts/run_ci.sh

docker-build:
	docker build -t rag-assistant:latest .

docker-up:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/
