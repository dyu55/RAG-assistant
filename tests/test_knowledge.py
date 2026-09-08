import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from rag_assistant.api import create_app
from rag_assistant.config import Settings
from rag_assistant.demo import DOCUMENTS, QUESTIONS
from rag_assistant.models import Claim, Draft, Question
from rag_assistant.providers import ProviderError
from rag_assistant.retrieval import graph_scores, graph_view
from rag_assistant.service import KnowledgeService, verify
from rag_assistant.text import load_pages, split_document


@pytest.fixture
def service(tmp_path):
    service = KnowledgeService(Settings(data_dir=tmp_path))
    for name, text in DOCUMENTS.items():
        service.ingest(name, text.encode())
    return service


@pytest.mark.parametrize("question,expected", QUESTIONS)
def test_retrieval_and_quotations(service, question, expected):
    answer = service.ask(Question(text=question))
    assert answer.status == "supported"
    assert expected in {e.filename for e in answer.evidence}
    assert all(
        claim.quote in next(e.text for e in answer.evidence if e.source_id == claim.source_id)
        for claim in answer.claims
    )
    assert not answer.cached


@pytest.mark.parametrize("mode", ["hybrid", "keyword", "vector", "graph"])
def test_each_retrieval_channel(service, mode):
    answer = service.ask(Question(text="How does Atlas handle Redis?", mode=mode))
    assert answer.status == "supported"
    assert all(e.relevance >= 0 and e.fusion_score > 0 for e in answer.evidence)
    if mode != "hybrid":
        assert all(e.channels == [mode] for e in answer.evidence)


def test_unknown_question_and_empty_library_abstain(service, tmp_path):
    assert service.ask(Question(text="What is Neptune's orbital period?")).status == "abstained"
    empty = KnowledgeService(Settings(data_dir=tmp_path / "empty"))
    answer = empty.ask(Question(text="What is Atlas?"))
    assert answer.status == "abstained" and answer.evidence == []


def test_cache_preserves_evidence_but_is_not_mutable(service):
    q = Question(text="How does Atlas handle Redis?")
    first = service.ask(q)
    original = first.evidence[0].text
    first.evidence[0].text = "corrupted by caller"
    second = service.ask(q)
    assert second.cached and second.id != first.id
    assert second.evidence[0].text == original
    assert second.claims and second.reason == first.reason
    assert len(service.store.history()) == 2


def test_replacement_and_deletion_invalidate_cache(service):
    q = Question(text="How does Atlas handle Redis?")
    first = service.ask(q)
    service.ingest(
        "Cache operations.md",
        b"Atlas reads from Postgres when Redis is unavailable. The retry policy uses exponential backoff.",
    )
    second = service.ask(q)
    assert not second.cached and second.revision > first.revision
    target = service.store.document("Cache operations.md")
    assert service.store.delete(target["id"])
    third = service.ask(q)
    assert not third.cached
    assert target["id"] not in {e.document_id for e in third.evidence}
    assert not service.store.delete(target["id"])


def test_duplicate_content_and_persistent_history(service):
    before = service.store.snapshot()[0]
    result = service.ingest("Cache operations.md", DOCUMENTS["Cache operations.md"].encode())
    assert result["unchanged"] and service.store.snapshot()[0] == before
    answer = service.ask(Question(text="How does Atlas use Postgres?"))
    restored = KnowledgeService(service.settings)
    assert restored.store.history()[0]["id"] == answer.id
    assert len(restored.store.list_documents()) == 4


def test_document_filter_does_not_leak_other_sources(service):
    doc = service.store.document("Reliability playbook.md")
    answer = service.ask(Question(text="Who coordinates recovery?", document_ids=[doc["id"]]))
    assert answer.evidence
    assert {e.document_id for e in answer.evidence} == {doc["id"]}
    missing = service.ask(Question(text="Atlas", document_ids=["unknown-id"]))
    assert not missing.evidence


def test_citation_fabrication_and_mismatched_quotes_are_rejected(service):
    answer = service.ask(Question(text="Atlas Redis outage"))
    source = answer.evidence[0]
    for claim in [
        Claim(text="Unsupported answer", source_id="S999", quote="Unsupported answer"),
        Claim(
            text="Atlas has infinite capacity.",
            source_id=source.source_id,
            quote="Atlas has infinite capacity.",
        ),
    ]:
        valid, reason = verify(Draft(claims=[claim]), answer.evidence)
        assert not valid and reason


def test_provider_failure_has_explicit_extractive_fallback(service, monkeypatch):
    service.settings.provider, service.settings.model = "ollama", "fake"

    def fail(*args):
        raise ProviderError("offline")

    monkeypatch.setattr(service.client, "generate", fail)
    answer = service.ask(Question(text="How does Atlas handle Redis?"))
    assert answer.provider == "extractive-fallback" and answer.warnings and answer.claims


def test_embedding_failure_falls_back_only_for_hybrid(service, monkeypatch):
    def fail(*args):
        raise ProviderError("offline")

    monkeypatch.setattr(service.client, "embed", fail)
    answer = service.ask(Question(text="How does Atlas handle Redis?"))
    assert answer.claims and answer.warnings
    assert all("vector" not in e.channels for e in answer.evidence)
    with pytest.raises(ProviderError):
        service.ask(Question(text="Atlas", mode="vector"))


def test_failed_ingestion_keeps_old_document_and_revision(service, monkeypatch):
    before = service.store.snapshot()

    def fail(*args):
        raise ProviderError("offline")

    monkeypatch.setattr(service.client, "embed", fail)
    with pytest.raises(ProviderError):
        service.ingest("Cache operations.md", b"Changed document content that cannot be embedded.")
    after = service.store.snapshot()
    assert before == after


def test_index_limit_is_transactional(tmp_path):
    service = KnowledgeService(Settings(data_dir=tmp_path, max_chunks=1))
    service.ingest("one.txt", b"The first document remains available.")
    with pytest.raises(ValueError, match="limit"):
        service.ingest("two.txt", b"The second document would exceed the limit.")
    assert [doc["filename"] for doc in service.store.list_documents()] == ["one.txt"]


def test_embedding_identity_and_dimensions_are_checked(service):
    service.settings.embedding_model = "another-model"
    service.settings.embedding_provider = "ollama"
    with pytest.raises(ValueError, match="Embedding model"):
        service.ask(Question(text="Atlas"))
    with pytest.raises(ValueError, match="Embedding model"):
        service.ingest("Cache operations.md", DOCUMENTS["Cache operations.md"].encode())


def test_changed_embedding_dimensions_cannot_corrupt_index(service, monkeypatch):
    before = service.store.snapshot()
    monkeypatch.setattr(service.client, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    with pytest.raises(ValueError, match="dimensions"):
        service.ingest("new.txt", b"Atlas has another useful document.")
    assert service.store.snapshot() == before


def test_chunk_configuration_change_reindexes_identical_content(tmp_path):
    original = KnowledgeService(Settings(data_dir=tmp_path))
    content = b"Atlas uses Redis to cache delivery status. " * 20
    original.ingest("notes.txt", content)
    revision = original.store.snapshot()[0]
    changed = KnowledgeService(Settings(data_dir=tmp_path, chunk_size=300, chunk_overlap=50))
    result = changed.ingest("notes.txt", content)
    assert not result["unchanged"] and changed.store.snapshot()[0] > revision
    assert result["chunks"] > 1


def test_parallel_imports_have_consistent_snapshots(tmp_path):
    service = KnowledgeService(Settings(data_dir=tmp_path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda i: service.ingest(
                    f"note-{i}.md", f"Atlas has document number {i}.".encode()
                ),
                range(8),
            )
        )
    revision, _, chunks = service.store.snapshot()
    assert len(results) == len(chunks) == revision == 8


def test_chunk_offsets_and_overlap():
    text = "Alpha beta gamma delta.\n" * 60
    chunks = split_document("note.txt", text.encode(), 180, 30)
    assert len(chunks) > 2
    assert all(text[c.start : c.end] == c.text for c in chunks)
    assert all(a.end > b.start for a, b in zip(chunks, chunks[1:], strict=False))
    assert chunks[-1].end == len(text)


def test_two_hop_graph_retrieves_linked_document():
    chunks = []
    for name, text in [
        ("a.txt", "[[Atlas]] works with [[Beacon]]."),
        ("b.txt", "[[Beacon]] works with [[Cobalt]]."),
        ("c.txt", "[[Cobalt]] works with [[Delta]]."),
    ]:
        chunks.extend(split_document(name, text.encode(), 900, 150))
    scores, paths = graph_scores("Atlas", chunks, max_hops=2)
    target = next(c for c in chunks if c.filename == "c.txt")
    assert target.id in scores
    assert paths[target.id] == ["atlas", "beacon", "cobalt"]
    assert graph_view(chunks)["edges"][0]["relation"] == "co-occurs"


@pytest.mark.parametrize(
    "filename,content",
    [
        ("a.exe", b"abc"),
        ("a.txt", b"\xff\xfe"),
        ("a.txt", b"binary\x00value"),
        ("empty.txt", b"   "),
        ("bad.pdf", b"not a PDF"),
    ],
)
def test_bad_documents_are_rejected(filename, content):
    with pytest.raises(ValueError):
        load_pages(filename, content)


def test_html_omits_scripts_and_blank_pdf_reports_ocr():
    pages = load_pages(
        "page.html", b"<h1>Atlas</h1><script>do not index</script><p>Uses Redis &amp; Postgres.</p>"
    )
    assert "do not index" not in pages[0][1]
    assert "Redis & Postgres" in pages[0][1]
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="OCR"):
        load_pages("scan.pdf", buffer.getvalue())


def test_real_pdf_text_can_be_indexed_and_cited(tmp_path):
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(600, 800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 50 740 Td (Atlas uses rolling deployments through the Gateway.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    service = KnowledgeService(Settings(data_dir=tmp_path))
    service.ingest("deployments.pdf", buffer.getvalue())
    answer = service.ask(Question(text="How does Atlas deploy?"))
    assert answer.status == "supported"
    assert answer.evidence[0].filename == "deployments.pdf" and answer.evidence[0].page == 1
    assert "rolling deployments" in answer.answer


def test_browser_api_document_and_answer_lifecycle(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        uploaded = client.post(
            "/api/documents",
            files={
                "file": (
                    "notes.md",
                    b"Atlas uses Redis as an optional cache. Postgres stores delivery orders.",
                )
            },
        )
        assert uploaded.status_code == 201
        answer = client.post("/api/ask", json={"text": "What stores delivery orders?"})
        assert answer.status_code == 200 and answer.json()["claims"]
        assert len(client.get("/api/history").json()) == 1
        assert client.delete("/api/documents/" + uploaded.json()["id"]).status_code == 200
        assert client.get("/api/health").json()["documents"] == 0


def test_api_rejects_invalid_input_and_cross_site_changes(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_upload_bytes=1024))) as client:
        assert client.post("/api/ask", json={"text": "   "}).status_code == 422
        assert (
            client.post("/api/documents", files={"file": ("big.txt", b"a" * 1025)}).status_code
            == 413
        )
        assert (
            client.post("/api/demo", headers={"Origin": "https://evil.example"}).status_code == 403
        )
        assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400
        assert client.post("/api/demo").status_code == 200
        assert client.get("/api/graph").json()["nodes"]
