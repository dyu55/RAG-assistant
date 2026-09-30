from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.crag import (
    decompose_into_knowledge_strips,
    evaluate_retrieval,
    generate_corrective_query,
    refine_knowledge_strips,
)
from rag_assistant.models import Evidence, Question
from rag_assistant.service import KnowledgeService


def _make_ev(source_id: str, text: str, relevance: float) -> Evidence:
    return Evidence(
        source_id=source_id,
        chunk_id=f"c_{source_id}",
        document_id="d1",
        filename="system.md",
        text=text,
        page=1,
        start=0,
        end=len(text),
        relevance=relevance,
        fusion_score=0.1,
        channels=["keyword"],
    )


def test_evaluate_retrieval_empty_and_disjoint():
    eval_empty = evaluate_retrieval("", [])
    assert eval_empty.action == "incorrect"
    assert eval_empty.confidence == 0.0

    ev = _make_ev("S1", "Random content about cooking and gardening.", 0.1)
    eval_disjoint = evaluate_retrieval("Atlas distributed architecture", [ev])
    assert eval_disjoint.action == "incorrect"
    assert eval_disjoint.confidence < 0.25


def test_evaluate_retrieval_correct_and_ambiguous():
    ev_high = _make_ev("S1", "Atlas architecture implements distributed cache replication.", 0.95)
    eval_correct = evaluate_retrieval("Atlas distributed cache", [ev_high])
    assert eval_correct.action == "correct"
    assert eval_correct.confidence >= 0.60

    ev_med = _make_ev("S1", "Atlas has some caching features.", 0.45)
    eval_ambiguous = evaluate_retrieval("Atlas distributed cache failover", [ev_med])
    assert eval_ambiguous.action in {"ambiguous", "incorrect"}


def test_generate_corrective_query():
    raw_query = "Please could you explain to me how does Atlas handle cache?"
    corrected = generate_corrective_query(raw_query, [])
    assert "please" not in corrected.lower()
    assert "explain" not in corrected.lower()
    assert "atlas" in corrected.lower()
    assert "cache" in corrected.lower()


def test_decompose_into_knowledge_strips():
    text = "First strip contains details. Second strip provides definitions. Third strip concludes."
    strips = decompose_into_knowledge_strips(text)
    assert len(strips) == 3
    assert "First strip contains details." in strips[0]


def test_refine_knowledge_strips():
    text = (
        "Atlas coordinates with Redis for distributed cache. "
        "The weather in Antarctica is extremely cold and windy. "
        "Redis provides automatic sub-second failover."
    )
    ev = _make_ev("S1", text, 0.9)
    refined = refine_knowledge_strips([ev], query="How does Atlas integrate with Redis?")
    assert len(refined) == 1
    # Relevant sentences kept in compressed_text
    assert "Atlas coordinates with Redis" in refined[0].compressed_text
    # Unrelated sentence removed
    assert "Antarctica" not in refined[0].compressed_text
    # Original text must remain unchanged for citation integrity
    assert ev.text == refined[0].text


def test_service_crag_end_to_end(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        provider="offline",
        crag_enabled=True,
        attention_reordering_enabled=True,
    )
    service = KnowledgeService(settings)
    doc_content = (
        b"Atlas maintains localized vector shards across multiple nodes.\n"
        b"Each shard can execute approximate nearest neighbor search independently.\n"
        b"Central coordinator aggregates candidate vectors and reranks them.\n"
    )
    service.ingest("guide.txt", doc_content)

    q = Question(text="Can you explain how Atlas maintains localized vector shards?")
    answer = service.ask(q)

    assert answer.status == "supported"
    assert len(answer.evidence) >= 1
    # Claims verified
    assert len(answer.claims) >= 1
    for claim in answer.claims:
        assert any(claim.quote in e.text for e in answer.evidence)
