from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from rag_assistant.config import Settings
from rag_assistant.models import Claim, Draft, Evidence, Question
from rag_assistant.service import KnowledgeService
from rag_assistant.speculative import (
    compute_groundedness,
    compute_inter_draft_consensus,
    compute_query_alignment,
    execute_speculative_rag,
    partition_evidence_subsets,
    synthesize_consensus_claims,
)


def make_evidence(
    source_id: str,
    text: str,
    doc_id: str = "doc1",
    chunk_id: str = "c1",
    relevance: float = 0.8,
) -> Evidence:
    return Evidence(
        source_id=source_id,
        chunk_id=chunk_id,
        document_id=doc_id,
        filename="test.txt",
        text=text,
        page=1,
        start=0,
        end=len(text),
        relevance=relevance,
        fusion_score=0.5,
        channels=["keyword"],
    )


def test_partition_evidence_subsets():
    # Empty and single
    assert partition_evidence_subsets([]) == []
    ev1 = make_evidence("S1", "First chunk", doc_id="d1", chunk_id="c1", relevance=0.9)
    assert len(partition_evidence_subsets([ev1])) == 1
    assert partition_evidence_subsets([ev1], max_subsets=1) == [[ev1]]

    # Multiple distinct documents
    ev2 = make_evidence("S2", "Second chunk", doc_id="d2", chunk_id="c2", relevance=0.8)
    ev3 = make_evidence("S3", "Third chunk", doc_id="d3", chunk_id="c3", relevance=0.7)
    subsets_multi = partition_evidence_subsets([ev1, ev2, ev3], max_subsets=2)
    assert len(subsets_multi) == 2
    # Combined evidence should preserve all elements
    all_ev = [e.source_id for sub in subsets_multi for e in sub]
    assert sorted(all_ev) == ["S1", "S2", "S3"]

    # Single document rank interleaving
    ev4 = make_evidence("S4", "Fourth chunk", doc_id="d1", chunk_id="c4", relevance=0.6)
    subsets_interleaved = partition_evidence_subsets([ev1, ev2, ev3, ev4], max_subsets=2)
    assert len(subsets_interleaved) == 2


def test_compute_groundedness():
    ev = make_evidence(
        "S1",
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        doc_id="d1",
        chunk_id="c1",
    )

    # Valid grounded claim
    valid_claim = Claim(
        text="FastAPI is a fast web framework for APIs.",
        quote="FastAPI is a modern, fast web framework for building APIs with Python.",
        source_id="S1",
    )
    draft_valid = Draft(claims=[valid_claim])
    assert compute_groundedness(draft_valid, [ev]) == 1.0

    # Hallucinated quote not in text
    fake_quote_claim = Claim(
        text="Django is older than FastAPI.",
        quote="Django was released in 2005.",
        source_id="S1",
    )
    draft_fake_quote = Draft(claims=[fake_quote_claim])
    assert compute_groundedness(draft_fake_quote, [ev]) == 0.0

    # Unknown source_id
    unknown_src_claim = Claim(
        text="FastAPI is modern.",
        quote="FastAPI is modern",
        source_id="S99",
    )
    assert compute_groundedness(Draft(claims=[unknown_src_claim]), [ev]) == 0.0

    # Empty draft
    assert compute_groundedness(Draft(claims=[]), [ev]) == 0.0


def test_compute_query_alignment():
    claim = Claim(
        text="PostgreSQL supports ACID compliant relational transactions.",
        quote="PostgreSQL supports ACID compliant relational transactions.",
        source_id="S1",
    )
    draft = Draft(claims=[claim])

    # Relevant query
    score_rel = compute_query_alignment(draft, "Does PostgreSQL support ACID transactions?")
    assert score_rel >= 0.5

    # Unrelated query
    score_unrel = compute_query_alignment(draft, "How to cook pasta bolognese?")
    assert score_unrel == 0.0

    # Empty query or draft
    assert compute_query_alignment(Draft(claims=[]), "PostgreSQL") == 0.0
    assert compute_query_alignment(draft, "") == 0.0


def test_compute_inter_draft_consensus():
    c1 = Claim(
        text="Redis is an in-memory key-value data store.",
        quote="Redis is an in-memory key-value data store.",
        source_id="S1",
    )
    c2 = Claim(
        text="Redis stores data in-memory as key-value pairs.",
        quote="Redis stores data in-memory as key-value pairs.",
        source_id="S2",
    )
    c3 = Claim(
        text="Quantum computers use qubits instead of bits.",
        quote="Quantum computers use qubits instead of bits.",
        source_id="S3",
    )

    d1 = Draft(claims=[c1])
    d2 = Draft(claims=[c2])
    d3 = Draft(claims=[c3])

    # Single draft
    assert compute_inter_draft_consensus([d1]) == [1.0]

    # Two agreeing drafts
    scores_agree = compute_inter_draft_consensus([d1, d2])
    assert scores_agree[0] > 0.4 and scores_agree[1] > 0.4

    # Two agreeing drafts and one divergent draft
    scores_three = compute_inter_draft_consensus([d1, d2, d3])
    assert scores_three[0] > scores_three[2]
    assert scores_three[1] > scores_three[2]

    # Draft with empty claims
    scores_empty = compute_inter_draft_consensus([d1, Draft(claims=[])])
    assert scores_empty[1] == 0.0


def test_synthesize_consensus_claims():
    ev1 = make_evidence("S1", "Docker containers package software with dependencies.")
    ev2 = make_evidence("S2", "Kubernetes orchestrates containerized workloads across clusters.")

    c1 = Claim(
        text="Docker containers package application software.",
        quote="Docker containers package software with dependencies.",
        source_id="S1",
    )
    c2 = Claim(
        text="Kubernetes manages containerized workloads across server clusters.",
        quote="Kubernetes orchestrates containerized workloads across clusters.",
        source_id="S2",
    )
    c3_dup = Claim(
        text="Docker containers package application software.",
        quote="Docker containers package software with dependencies.",
        source_id="S1",
    )

    d1 = Draft(claims=[c1])
    d2 = Draft(claims=[c2, c3_dup])

    ev_map = {"S1": ev1, "S2": ev2}
    synth = synthesize_consensus_claims([d1, d2], [0.9, 0.85], ev_map, max_claims=5)
    # Deduplication should preserve c1 and c2, omitting duplicate c3_dup
    assert len(synth.claims) == 2
    assert {c.source_id for c in synth.claims} == {"S1", "S2"}


def test_execute_speculative_rag():
    ev1 = make_evidence(
        "S1",
        "RAFT combines domain-specific RAG with supervised fine-tuning.",
        doc_id="d1",
        chunk_id="c1",
        relevance=0.9,
    )
    ev2 = make_evidence(
        "S2",
        "Supervised fine-tuning teaches models to cite relevant passages while ignoring distractors.",
        doc_id="d2",
        chunk_id="c2",
        relevance=0.85,
    )

    def mock_drafter(query: str, subset: list[Evidence]) -> Draft:
        claims = []
        for e in subset:
            if "RAFT" in e.text:
                claims.append(
                    Claim(
                        text="RAFT combines RAG with fine-tuning.",
                        quote="RAFT combines domain-specific RAG with supervised fine-tuning.",
                        source_id=e.source_id,
                    )
                )
            elif "Supervised" in e.text:
                claims.append(
                    Claim(
                        text="Fine-tuning teaches models to cite relevant passages.",
                        quote=(
                            "Supervised fine-tuning teaches models to cite relevant passages "
                            "while ignoring distractors."
                        ),
                        source_id=e.source_id,
                    )
                )
        return Draft(claims=claims)

    report, selected = execute_speculative_rag(
        query="What is RAFT and how does fine-tuning help?",
        evidence=[ev1, ev2],
        drafter_fn=mock_drafter,
        max_subsets=2,
    )

    assert report.enabled is True
    assert report.num_subsets == 2
    assert len(report.drafts) == 2
    assert report.selection_strategy in {"consensus_synthesis", "best_candidate"}
    assert len(selected.claims) >= 1
    assert report.consensus_score >= 0.0


def test_execute_speculative_rag_fallback_on_ungrounded():
    ev1 = make_evidence("S1", "Actual truth in text 1.")
    ev2 = make_evidence("S2", "Actual truth in text 2.")

    def bad_drafter(query: str, subset: list[Evidence]) -> Draft:
        if len(subset) == 1:
            # Candidate drafts return completely ungrounded claims
            return Draft(
                claims=[
                    Claim(text="Hallucinated fact.", quote="Nonexistent quote.", source_id="S99")
                ]
            )
        # Fallback receives full context and produces valid grounded claim
        return Draft(
            claims=[
                Claim(
                    text="Actual truth in text 1.", quote="Actual truth in text 1.", source_id="S1"
                )
            ]
        )

    report, selected = execute_speculative_rag(
        query="Tell me the truth",
        evidence=[ev1, ev2],
        drafter_fn=bad_drafter,
        max_subsets=2,
    )

    assert report.selection_strategy == "fallback_single_pass"
    assert len(selected.claims) == 1
    assert selected.claims[0].source_id == "S1"


def test_execute_speculative_rag_empty_and_single():
    report_empty, draft_empty = execute_speculative_rag(
        query="Empty query",
        evidence=[],
        drafter_fn=lambda q, s: Draft(claims=[]),
    )
    assert report_empty.num_subsets == 0
    assert len(draft_empty.claims) == 0

    ev = make_evidence("S1", "Sole evidence passage.")
    report_single, draft_single = execute_speculative_rag(
        query="Query",
        evidence=[ev],
        drafter_fn=lambda q, s: Draft(
            claims=[
                Claim(
                    text="Sole evidence passage.",
                    quote="Sole evidence passage.",
                    source_id="S1",
                )
            ]
        ),
    )
    assert report_single.num_subsets == 1
    assert report_single.selection_strategy == "single_pass"
    assert len(draft_single.claims) == 1


def test_service_speculative_rag_integration():
    with TemporaryDirectory() as tmp_dir:
        settings = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            speculative_rag_enabled=True,
            speculative_max_subsets=2,
            chunk_size=150,
            chunk_overlap=30,
        )
        service = KnowledgeService(settings)

        doc_text = (
            "Kafka is a distributed event streaming platform used for high-performance data pipelines.\n\n"
            "Kafka provides durability and fault tolerance through topic partitioning and replication across brokers.\n\n"
            "Consumers read messages sequentially from partitions using offset checkpoints."
        )
        service.ingest("kafka_guide.txt", doc_text.encode("utf-8"))

        # Question triggering retrieval of multiple chunks
        answer = service.ask(
            Question(
                text="What is Kafka and how does it achieve fault tolerance?",
                mode="keyword",
                top_k=4,
            )
        )

        assert answer.status == "supported"
        assert answer.speculative is not None
        assert answer.speculative.enabled is True
        assert answer.speculative.num_subsets >= 1
        assert len(answer.claims) > 0
        assert any(
            c in answer.corrections
            for c in [
                "speculative_rag_best_candidate_selection",
                "speculative_rag_consensus_synthesis",
            ]
        ) or answer.speculative.selection_strategy in {
            "best_candidate",
            "consensus_synthesis",
            "single_pass",
        }

        # Disabled setting
        settings_disabled = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            speculative_rag_enabled=False,
        )
        service_disabled = KnowledgeService(settings_disabled)
        answer_disabled = service_disabled.ask(
            Question(text="What is Kafka?", mode="keyword", top_k=2)
        )
        assert answer_disabled.speculative is None
