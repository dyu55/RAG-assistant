from __future__ import annotations

import time
from unittest.mock import Mock

from fastapi.testclient import TestClient

from rag_assistant.api import create_app
from rag_assistant.cache import SemanticCache, cosine_similarity
from rag_assistant.config import Settings
from rag_assistant.models import Answer, Claim, Draft, Question
from rag_assistant.service import KnowledgeService


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_semantic_cache_exact_and_lru():
    cache = SemanticCache(ttl=60.0, threshold=0.90, max_entries=2)
    q = Question(text="How does Atlas scale?")
    ans = Answer(
        id="a1",
        question="How does Atlas scale?",
        answer="Atlas scales horizontally.",
        claims=[],
        evidence=[],
        status="supported",
        reason="ok",
        provider="offline",
        retrieval_mode="hybrid",
        confidence=1.0,
        revision=1,
    )

    cache.put(q, revision=1, provider="offline", model="", answer=ans, vector=[1.0, 0.0])

    # Exact hit with case variation and spaces
    q_case = Question(text="  how DOES atlas scale?  ")
    hit = cache.get_exact(q_case, revision=1, provider="offline", model="")
    assert hit is not None
    assert hit.cached is True
    assert hit.id != ans.id
    assert hit.answer == ans.answer

    # Miss on different revision
    assert cache.get_exact(q, revision=2, provider="offline", model="") is None

    # Miss on different provider
    assert cache.get_exact(q, revision=1, provider="openai", model="") is None


def test_semantic_cache_vector_similarity():
    cache = SemanticCache(ttl=60.0, threshold=0.90, max_entries=10)
    q_base = Question(text="How does Atlas handle Redis?")
    ans = Answer(
        id="a1",
        question=q_base.text,
        answer="Atlas coordinates with Redis.",
        claims=[],
        evidence=[],
        status="supported",
        reason="ok",
        provider="offline",
        retrieval_mode="hybrid",
        confidence=1.0,
        revision=1,
    )
    cache.put(q_base, revision=1, provider="offline", model="", answer=ans, vector=[0.9, 0.1])

    # High similarity vector (similarity > 0.90)
    q_similar = Question(text="Atlas Redis coordination details?")
    sem_hit = cache.get_semantic(
        q_similar,
        revision=1,
        provider="offline",
        model="",
        vector=[0.89, 0.12],
        threshold=0.90,
    )
    assert sem_hit is not None
    assert sem_hit.cached is True
    assert any("semantic cache" in w for w in sem_hit.warnings)

    # Low similarity vector
    q_diff = Question(text="Explain Postgres replication")
    sem_miss = cache.get_semantic(
        q_diff,
        revision=1,
        provider="offline",
        model="",
        vector=[0.1, 0.9],
        threshold=0.90,
    )
    assert sem_miss is None


def test_semantic_cache_ttl_expiry():
    cache = SemanticCache(ttl=0.05, threshold=0.90, max_entries=5)
    q = Question(text="Temporary question?")
    ans = Answer(
        id="a1",
        question=q.text,
        answer="Temporary answer.",
        claims=[],
        evidence=[],
        status="supported",
        reason="ok",
        provider="offline",
        retrieval_mode="hybrid",
        confidence=1.0,
        revision=1,
    )
    cache.put(q, revision=1, provider="offline", model="", answer=ans)
    assert cache.get_exact(q, revision=1, provider="offline", model="") is not None

    time.sleep(0.06)
    assert cache.get_exact(q, revision=1, provider="offline", model="") is None


def test_service_semantic_cache_end_to_end(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", provider="openai", model="mock-model")
    mock_client = Mock()
    # Embed returns distinct vectors
    mock_client.embed.side_effect = lambda texts: [[0.8, 0.2] for _ in texts]
    mock_client.generate.return_value = Draft(
        claims=[
            Claim(
                source_id="S1",
                text="Atlas handles Redis caching efficiently.",
                quote="Atlas coordinates with Redis cluster.",
            )
        ]
    )

    service = KnowledgeService(settings, client=mock_client)
    service.ingest(
        "cache.md", b"Atlas coordinates with Redis cluster. Redis stores distributed sessions."
    )

    q1 = Question(text="How does Atlas handle Redis?")
    ans1 = service.ask(q1)
    assert ans1.status == "supported"
    assert not ans1.cached

    # Second call with identical query hits exact/fast cache
    ans2 = service.ask(q1)
    assert ans2.cached is True
    assert ans2.id != ans1.id

    # Third call with similar query hits semantic cache
    q3 = Question(text="How does Atlas manage Redis caching?")
    ans3 = service.ask(q3)
    assert ans3.cached is True

    # Test API stats & clear
    app = create_app(settings, service)
    with TestClient(app) as client:
        stats = client.get("/api/cache/stats").json()
        assert stats["total_hits"] >= 2
        assert stats["hit_rate"] > 0.0

        res_clear = client.post("/api/cache/clear")
        assert res_clear.status_code == 200
        stats_cleared = client.get("/api/cache/stats").json()
        assert stats_cleared["size"] == 0
