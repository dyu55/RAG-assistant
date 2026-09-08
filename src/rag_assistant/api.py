from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .config import Settings
from .demo import DOCUMENTS
from .models import Answer, Question
from .providers import ProviderError
from .retrieval import graph_view
from .service import KnowledgeService


def create_app(
    settings: Settings | None = None, service: KnowledgeService | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or KnowledgeService(settings)
    app = FastAPI(title="RAG Assistant", version=__version__)
    app.state.service = service
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )

    @app.middleware("http")
    async def browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if (
                origin and urlsplit(origin).netloc != request.headers.get("host")
            ) or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(
                    {"detail": "Cross-origin changes are not allowed"}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path in {"/docs", "/redoc"}:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"
            )
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ProviderError)
    async def unavailable(request: Request, exc: ProviderError):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/api/health")
    def health():
        documents = service.store.list_documents()
        return {
            "status": "ok",
            "version": __version__,
            "documents": len(documents),
            "chunks": sum(row["chunks"] for row in documents),
            "provider": settings.provider,
            "embedding_provider": settings.embedding_provider,
            "max_upload_bytes": settings.max_upload_bytes,
        }

    @app.get("/api/documents")
    def documents():
        return service.store.list_documents()

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile):
        try:
            content = await file.read(settings.max_upload_bytes + 1)
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(413, "Document exceeds the upload size limit")
            return await run_in_threadpool(service.ingest, file.filename or "", content)
        finally:
            await file.close()

    @app.delete("/api/documents/{document_id}")
    def delete(document_id: str):
        if not service.store.delete(document_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": document_id}

    @app.post("/api/demo")
    def demo():
        return [service.ingest(name, text.encode()) for name, text in DOCUMENTS.items()]

    @app.post("/api/ask", response_model=Answer)
    def ask(question: Question):
        return service.ask(question)

    @app.get("/api/history")
    def history():
        return service.store.history()

    @app.get("/api/graph")
    def graph():
        _, _, chunks = service.store.snapshot()
        return graph_view(chunks)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    return app
