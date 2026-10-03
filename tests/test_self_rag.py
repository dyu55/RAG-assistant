from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.models import Claim, Draft, Evidence, Question
from rag_assistant.self_rag import (
    critique_answer_utility,
    critique_claim_support,
    critique_passage_relevance,
    critique_retrieval_need,
    execute_self_reflection,
)
from rag_assistant.service import KnowledgeService


def _create_ev(source_id: str, text: str, relevance: float) -> Evidence:
    return Evidence(
        source_id=source_id,
        chunk_id=f"c_{source_id}",
        document_id="d1",
        filename="doc.txt",
        text=text,
        page=1,
        start=0,
        end=len(text),
        relevance=relevance,
        fusion_score=0.1,
        channels=["keyword"],
    )


def test_critique_retrieval_need():
    assert critique_retrieval_need("Hello") == "[NoRetrieve]"
    assert critique_retrieval_need("Good morning") == "[NoRetrieve]"
    assert critique_retrieval_need("Thanks!") == "[NoRetrieve]"
    assert critique_retrieval_need("How does Atlas replicate caches?") == "[Retrieve]"


def test_critique_passage_relevance():
    ev_rel = _create_ev("S1", "Atlas integrates with Redis cluster for high throughput cache.", 0.8)
    assert critique_passage_relevance(ev_rel, "How does Atlas use Redis cache?") == "[IsRel]"

    ev_not_rel = _create_ev(
        "S2", "Weather patterns in the Sahara desert during summer months.", 0.05
    )
    assert critique_passage_relevance(ev_not_rel, "How does Atlas use Redis cache?") == "[NotRel]"


def test_critique_claim_support_levels():
    passage = "Atlas uses Redis for distributed session storage and caching."
    ev = _create_ev("S1", passage, 0.9)
    ev_map = {"S1": ev}

    # Fully supported
    c_full = Claim(
        text="Atlas uses Redis for distributed session storage and caching",
        source_id="S1",
        quote="Atlas uses Redis for distributed session storage and caching.",
    )
    tok_full, score_full, _ = critique_claim_support(c_full, ev_map)
    assert tok_full == "[IsSup:fully]"
    assert score_full >= 0.70

    # Partially supported
    c_part = Claim(
        text="Atlas uses Redis cluster storage with automatic failover mechanism",
        source_id="S1",
        quote="Atlas uses Redis for distributed session storage and caching.",
    )
    tok_part, _, _ = critique_claim_support(c_part, ev_map)
    assert tok_part in {"[IsSup:partial]", "[IsSup:fully]"}

    # Hallucinated quote
    c_hallucinated = Claim(
        text="Atlas connects to MongoDB database",
        source_id="S1",
        quote="Atlas connects to MongoDB database.",
    )
    tok_none, score_none, _ = critique_claim_support(c_hallucinated, ev_map)
    assert tok_none == "[IsSup:none]"
    assert score_none == 0.0


def test_critique_answer_utility():
    assert critique_answer_utility("question", "", [], "abstained") == 1
    c1 = Claim(text="Claim 1", source_id="S1", quote="Quote 1 here.")
    c2 = Claim(text="Claim 2", source_id="S1", quote="Quote 2 here.")
    utility = critique_answer_utility(
        "query about topic", "Atlas provides query about topic with details", [c1, c2], "supported"
    )
    assert utility >= 4


def test_execute_self_reflection_closed_loop_healing():
    passage = "Atlas provides high-availability cache replication through Beacon gateway."
    ev = _create_ev("S1", passage, 0.95)

    valid_claim = Claim(
        text="Atlas provides high-availability cache replication through Beacon gateway",
        source_id="S1",
        quote="Atlas provides high-availability cache replication through Beacon gateway.",
    )
    hallucinated_claim = Claim(
        text="Atlas supports quantum computing acceleration",
        source_id="S1",
        quote="Quantum computing acceleration in cloud data centers.",
    )
    draft = Draft(claims=[valid_claim, hallucinated_claim])

    report, healed_draft, was_healed = execute_self_reflection(
        "How does Atlas replicate cache?", [ev], draft, "supported"
    )

    assert was_healed is True
    assert report.healed is True
    # Hallucinated claim pruned, valid claim preserved
    assert len(healed_draft.claims) == 1
    assert healed_draft.claims[0].text == valid_claim.text
    assert report.retrieve_decision == "[Retrieve]"
    assert report.passage_relevance["S1"] == "[IsRel]"


def test_service_self_rag_end_to_end(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        provider="offline",
        self_rag_enabled=True,
    )
    service = KnowledgeService(settings)
    doc_content = (
        b"Atlas maintains localized vector shards across multiple nodes.\n"
        b"Each shard executes nearest neighbor search independently.\n"
    )
    service.ingest("guide.txt", doc_content)

    q = Question(text="How does Atlas maintain localized vector shards?")
    answer = service.ask(q)

    assert answer.status == "supported"
    assert answer.reflection is not None
    assert answer.reflection.retrieve_decision == "[Retrieve]"
    assert answer.reflection.utility_score >= 3
    assert len(answer.reflection.claim_support) >= 1
    assert any(
        c["token"] in {"[IsSup:fully]", "[IsSup:partial]"} for c in answer.reflection.claim_support
    )
