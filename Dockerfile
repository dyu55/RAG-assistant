FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_DATA_DIR=/data
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home appuser && mkdir /data && chown appuser /data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["uvicorn", "rag_assistant.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
