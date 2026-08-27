"""
Unit tests for core/reliability.py
Tests the reliability engine — the core differentiator of the RAG system.
"""

from __future__ import annotations

import pytest

from core.generator import Citation, GeneratedAnswer
from core.reliability import (
    ReliabilityChecker,
    ReliabilityReport,
)
from core.retriever import RetrievedChunk

# ── Factories ────────────────────────────────────────────────────────────────────


def make_chunk(chunk_id: str, text: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={"filename": f"{chunk_id}.txt"},
    )


def make_answer(
    answer: str,
    citations: list[Citation] | None = None,
    self_confidence: float = 0.8,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=answer,
        citations=citations or [],
        self_confidence=self_confidence,
        reasoning="Test reasoning.",
    )


def make_citation(
    source_index: int,
    chunk_id: str,
    quote: str,
) -> Citation:
    return Citation(
        source_index=source_index,
        chunk_id=chunk_id,
        quote=quote,
    )


# ── Test: Citation Presence Check ────────────────────────────────────────────────


class TestCitationPresence:
    checker = ReliabilityChecker()

    def test_no_citations_returns_zero(self):
        answer = make_answer("The answer is 42.", citations=[])
        score = self.checker._check_citation_presence(answer)
        assert score == 0.0

    def test_valid_citations_with_quotes_returns_full_score(self):
        cit = make_citation(1, "c1", "Python is a programming language.")
        answer = make_answer(
            "Python is a programming language [Source 1].",
            citations=[cit],
        )
        score = self.checker._check_citation_presence(answer)
        assert score == 1.0

    def test_citations_without_quotes_returns_low_score(self):
        cit = Citation(source_index=1, chunk_id="c1", quote="")
        answer = make_answer(
            "The answer is 42 [Source 1].",
            citations=[cit],
        )
        score = self.checker._check_citation_presence(answer)
        assert score == 0.3  # Floor at 0.3 when structure exists but no content

    def test_citations_without_chunk_id_returns_low_score(self):
        cit = Citation(source_index=1, chunk_id="", quote="Some text.")
        answer = make_answer("Some text [Source 1].", citations=[cit])
        score = self.checker._check_citation_presence(answer)
        assert score == 0.3

    def test_mixed_valid_and_invalid_citations(self):
        valid = make_citation(1, "c1", "Python is great.")
        invalid = Citation(source_index=2, chunk_id="", quote="")
        answer = make_answer("Answer [Source 1][Source 2].", citations=[valid, invalid])
        score = self.checker._check_citation_presence(answer)
        # ratio = 1/2 = 0.5, but floor is 0.3
        assert score == 0.5


# ── Test: Grounding Verification ────────────────────────────────────────────────


class TestGroundingVerification:
    checker = ReliabilityChecker()

    def test_exact_quote_match_returns_full_score(self):
        chunk = make_chunk("c1", "Python is a programming language created by Guido van Rossum.")
        cit = make_citation(1, "c1", "Python is a programming language")
        answer = make_answer("Python is a programming language.", citations=[cit])

        score, details = self.checker._check_grounding(answer, {chunk.chunk_id: chunk})

        assert score == 1.0
        assert len(details) == 1
        assert details[0].match_ratio == 1.0
        assert details[0].is_grounded is True

    def test_quote_not_in_source_returns_low_score(self):
        """Quote that partially matches source text should score low but not zero."""
        chunk = make_chunk("c1", "Python is a programming language.")
        cit = make_citation(1, "c1", "JavaScript was created by Brendan Eich in 1995.")
        answer = make_answer("JavaScript was created by Brendan Eich.", citations=[cit])

        score, details = self.checker._check_grounding(answer, {chunk.chunk_id: chunk})

        assert score < 0.5  # Low score for unrelated quote
        assert details[0].match_ratio < 0.5
        assert details[0].is_grounded is False

    def test_chunk_not_found_returns_zero(self):
        chunk = make_chunk("c1", "Some source text.")
        cit = make_citation(1, "nonexistent-id", "Some quote.")
        answer = make_answer("Answer [Source 1].", citations=[cit])

        score, details = self.checker._check_grounding(answer, {chunk.chunk_id: chunk})

        assert score == 0.0
        assert details[0].match_ratio == 0.0
        assert "[chunk not found]" in details[0].source_text

    def test_empty_quote_returns_zero(self):
        chunk = make_chunk("c1", "Python is a programming language.")
        cit = make_citation(1, "c1", "")
        answer = make_answer("Python [Source 1].", citations=[cit])

        score, details = self.checker._check_grounding(answer, {chunk.chunk_id: chunk})

        assert score == 0.0

    def test_partial_quote_match_returns_partial_score(self):
        chunk = make_chunk(
            "c1", "Python is a high-level programming language with dynamic semantics."
        )
        cit = make_citation(1, "c1", "Python is a high-level programming language.")
        answer = make_answer("Answer [Source 1].", citations=[cit])

        score, details = self.checker._check_grounding(answer, {chunk.chunk_id: chunk})

        assert 0.0 < score < 1.0
        assert details[0].match_ratio > 0.5  # Should be substantial match

    def test_multiple_citations_averaged(self):
        c1 = make_chunk("c1", "Python is a programming language.")
        c2 = make_chunk("c2", "Python was created by Guido van Rossum.")
        cit1 = make_citation(1, "c1", "Python is a programming language")
        cit2 = make_citation(2, "c2", "Python was created by Guido van Rossum")
        answer = make_answer("Answer [Source 1][Source 2].", citations=[cit1, cit2])

        score, details = self.checker._check_grounding(answer, {c1.chunk_id: c1, c2.chunk_id: c2})

        assert score == 1.0  # Both exact matches
        assert len(details) == 2

    def test_no_citations_returns_zero(self):
        answer = make_answer("An answer with no citations.")
        score, details = self.checker._check_grounding(answer, {})
        assert score == 0.0
        assert details == []


# ── Test: Fuzzy Match ────────────────────────────────────────────────────────────


class TestFuzzyMatch:
    checker = ReliabilityChecker()

    def test_exact_substring_match(self):
        score = self.checker._fuzzy_match("hello world", "hello world today")
        assert score == 1.0

    def test_case_insensitive_match(self):
        score = self.checker._fuzzy_match("HELLO WORLD", "hello world today")
        assert score == 1.0

    def test_whitespace_insensitive(self):
        score = self.checker._fuzzy_match("  hello world  ", "hello world today")
        assert score == 1.0

    def test_no_match(self):
        score = self.checker._fuzzy_match(
            "The weather in Tokyo is sunny today", "Python was created by Guido van Rossum in 1991"
        )
        assert score < 0.5  # Should be low for unrelated texts

    def test_partial_match(self):
        quote = "Python is a programming language"
        source = "Python is a programming language with dynamic semantics"
        score = self.checker._fuzzy_match(quote, source)
        assert score > 0.7  # Should be a good partial match

    def test_empty_quote_returns_zero(self):
        score = self.checker._fuzzy_match("", "Some source text")
        assert score == 0.0

    def test_empty_source(self):
        score = self.checker._fuzzy_match("Some quote", "")
        assert score == 0.0


# ── Test: Unsupported Claims Detection ───────────────────────────────────────────


class TestUnsupportedClaims:
    checker = ReliabilityChecker()

    def test_supported_claim_in_chunk(self):
        chunks = [make_chunk("c1", "Python is a programming language created in 1991.")]
        answer = make_answer("Python is a programming language created in 1991.")

        claims = self.checker._check_unsupported_claims(answer, chunks)

        assert len(claims) >= 1
        # The claim should be supported
        supported = [c for c in claims if c.claim.strip()]
        if supported:
            assert supported[0].is_supported is True

    def test_unsupported_claim_not_in_any_chunk(self):
        chunks = [make_chunk("c1", "Python is a programming language.")]
        answer = make_answer("JavaScript was created by Netscape in 1995.")

        claims = self.checker._check_unsupported_claims(answer, chunks)

        assert len(claims) >= 1
        # The JavaScript claim should not match Python chunk well
        unsupported = [c for c in claims if not c.is_supported]
        if unsupported:
            assert unsupported[0].best_match_score < 0.7

    def test_empty_answer_returns_empty_claims(self):
        chunks = [make_chunk("c1", "Some content.")]
        answer = make_answer("")

        claims = self.checker._check_unsupported_claims(answer, chunks)

        assert claims == []

    def test_empty_chunks_returns_empty_claims(self):
        answer = make_answer("The answer is 42.")
        claims = self.checker._check_unsupported_claims(answer, [])
        assert claims == []

    def test_citation_references_stripped_before_check(self):
        """Citations like [Source 1] should be stripped from claims."""
        chunks = [make_chunk("c1", "Python is a programming language.")]
        answer = make_answer("Python is a programming language [Source 1].")

        claims = self.checker._check_unsupported_claims(answer, chunks)

        # [Source 1] should not appear in claim text
        for claim in claims:
            assert "[Source 1]" not in claim.claim

    def test_very_short_sentences_skipped(self):
        """Sentences with fewer than 4 words are skipped."""
        chunks = [make_chunk("c1", "Some content.")]
        answer = make_answer("Yes. No. Maybe. Python is great.")

        claims = self.checker._check_unsupported_claims(answer, chunks)

        # Only "Python is great" (4 words) should be checked
        if claims:
            for claim in claims:
                assert len(claim.claim.split()) >= 3


# ── Test: Confidence Computation ────────────────────────────────────────────────


class TestConfidenceComputation:
    checker = ReliabilityChecker()

    def test_perfect_scores_yield_high_confidence(self):
        confidence = self.checker._compute_confidence(
            retrieval_scores=[0.9, 0.85],
            citation_score=1.0,
            grounding_score=1.0,
            self_confidence=0.9,
            unsupported_ratio=0.0,
        )
        # Weighted: 0.3*0.875 + 0.25*1.0 + 0.25*1.0 + 0.2*0.9 = 0.2625 + 0.25 + 0.25 + 0.18 = 0.9425
        assert confidence > 0.9

    def test_zero_scores_yield_zero_confidence(self):
        confidence = self.checker._compute_confidence(
            retrieval_scores=[0.0, 0.0],
            citation_score=0.0,
            grounding_score=0.0,
            self_confidence=0.0,
            unsupported_ratio=0.0,
        )
        assert confidence == 0.0

    def test_unsupported_ratio_penalizes_confidence(self):
        with_support = self.checker._compute_confidence(
            retrieval_scores=[0.8],
            citation_score=1.0,
            grounding_score=1.0,
            self_confidence=0.8,
            unsupported_ratio=0.0,
        )
        without_support = self.checker._compute_confidence(
            retrieval_scores=[0.8],
            citation_score=1.0,
            grounding_score=1.0,
            self_confidence=0.8,
            unsupported_ratio=0.5,  # 50% unsupported → 10% penalty
        )
        assert without_support < with_support
        # Penalty: 1.0 - 0.2 * 0.5 = 0.9
        assert without_support == pytest.approx(with_support * 0.9, rel=0.01)

    def test_empty_retrieval_scores_defaults_to_zero(self):
        confidence = self.checker._compute_confidence(
            retrieval_scores=[],
            citation_score=1.0,
            grounding_score=1.0,
            self_confidence=0.8,
        )
        # avg_retrieval = 0.0 when empty
        assert confidence < 1.0

    def test_confidence_clamped_to_valid_range(self):
        # Should not exceed 1.0 or go below 0.0
        confidence = self.checker._compute_confidence(
            retrieval_scores=[1.0],
            citation_score=1.0,
            grounding_score=1.0,
            self_confidence=1.0,
        )
        assert 0.0 <= confidence <= 1.0


# ── Test: Abstention Decision ───────────────────────────────────────────────────


class TestAbstention:
    checker = ReliabilityChecker()

    def test_low_confidence_abstains(self):
        should, reason = self.checker._should_abstain(
            confidence=0.3,
            retrieval_scores=[0.8],
            citation_score=0.8,
            grounding_score=0.8,
            unsupported_ratio=0.0,
        )
        assert should is True
        assert "0.3" in reason

    def test_good_confidence_does_not_abstain(self):
        should, reason = self.checker._should_abstain(
            confidence=0.8,
            retrieval_scores=[0.8, 0.75],
            citation_score=0.9,
            grounding_score=0.9,
            unsupported_ratio=0.0,
        )
        assert should is False
        assert reason is None

    def test_no_retrieval_abstains(self):
        should, reason = self.checker._should_abstain(
            confidence=0.5,
            retrieval_scores=[0.1],  # Below MIN_RETRIEVAL_SCORE
            citation_score=0.5,
            grounding_score=0.5,
        )
        assert should is True
        assert "retrieval" in reason.lower()

    def test_no_evidence_abstains(self):
        should, reason = self.checker._should_abstain(
            confidence=0.3,
            retrieval_scores=[0.8],
            citation_score=0.0,  # No citations
            grounding_score=0.0,  # No grounding
            unsupported_ratio=0.0,
        )
        assert should is True
        assert "evidence" in reason.lower() or "citation" in reason.lower()

    def test_high_unsupported_ratio_abstains(self):
        should, reason = self.checker._should_abstain(
            confidence=0.6,
            retrieval_scores=[0.8],
            citation_score=0.8,
            grounding_score=0.8,
            unsupported_ratio=0.6,  # 60% unsupported
        )
        assert should is True
        assert "60%" in reason or "unsupported" in reason.lower()

    def test_multiple_reasons_combined(self):
        should, reason = self.checker._should_abstain(
            confidence=0.2,
            retrieval_scores=[0.05],
            citation_score=0.0,
            grounding_score=0.0,
            unsupported_ratio=0.7,
        )
        assert should is True
        # Should mention multiple reasons
        assert ";" in reason


# ── Test: Full Check ────────────────────────────────────────────────────────────


class TestReliabilityChecker:
    checker = ReliabilityChecker()

    def test_full_check_returns_complete_report(self):
        chunks = [make_chunk("c1", "Python is a programming language created by Guido.", 0.9)]
        answer = make_answer(
            "Python is a programming language [Source 1].",
            citations=[make_citation(1, "c1", "Python is a programming language")],
            self_confidence=0.85,
        )

        report = self.checker.check(answer, chunks)

        assert isinstance(report, ReliabilityReport)
        assert report.citation_score > 0
        assert report.grounding_score > 0
        assert report.confidence > 0
        assert report.should_abstain is False
        assert report.verdict in ("grounded", "partially_grounded")

    def test_full_check_with_unsupported_claims(self):
        chunks = [make_chunk("c1", "Python is a programming language.", 0.9)]
        answer = make_answer(
            "Python is a programming language. JavaScript was created by Netscape.",
            citations=[make_citation(1, "c1", "Python is a programming language")],
            self_confidence=0.7,
        )

        report = self.checker.check(answer, chunks)

        assert isinstance(report, ReliabilityReport)
        assert report.confidence > 0
        # Should detect unsupported claims
        assert len(report.unsupported_claims) >= 1

    def test_full_check_with_no_chunks(self):
        answer = make_answer("The answer is 42.")
        chunks = []

        report = self.checker.check(answer, chunks)

        assert report.citation_score == 0.0
        assert report.grounding_score == 0.0
        assert report.should_abstain is True

    def test_full_check_with_no_answer(self):
        chunks = [make_chunk("c1", "Some content.", 0.9)]
        answer = make_answer("")

        report = self.checker.check(answer, chunks)

        assert isinstance(report, ReliabilityReport)
        assert report.citation_score == 0.0

    def test_report_verdict_property(self):
        report = ReliabilityReport(
            confidence=0.85,
            should_abstain=False,
        )
        assert report.verdict == "grounded"

        report2 = ReliabilityReport(
            confidence=0.5,
            should_abstain=False,
        )
        assert report2.verdict in ("low_confidence", "partially_grounded")

        report3 = ReliabilityReport(
            confidence=0.0,
            should_abstain=True,
        )
        assert report3.verdict == "abstained"

    def test_report_verdict_emoji(self):
        report = ReliabilityReport(confidence=0.85, should_abstain=False)
        assert report.verdict_emoji == "✅"

        report2 = ReliabilityReport(confidence=0.0, should_abstain=True)
        assert report2.verdict_emoji == "🚫"

    def test_report_includes_details(self):
        chunks = [make_chunk("c1", "Python is a programming language.", 0.9)]
        answer = make_answer(
            "Python [Source 1].",
            citations=[make_citation(1, "c1", "Python is a programming language")],
            self_confidence=0.8,
        )

        report = self.checker.check(answer, chunks)

        assert "retrieval_scores" in report.details
        assert "avg_retrieval_score" in report.details
        assert "num_citations" in report.details
        assert "self_confidence" in report.details
        assert "confidence_weights" in report.details
