"""
Reliability Engine.
The core differentiator of this project. Performs post-generation checks to ensure
answer quality, detect hallucinations, and decide when to abstain.

Checks:
1. Citation Presence: Does the answer actually reference retrieved sources?
2. Grounding Verification: Do the cited quotes exist in the source chunks?
3. Confidence Scoring: Weighted aggregate of multiple quality signals.
4. Abstention Decision: Should the system refuse to answer?
"""
from __future__ import annotations

import re
import logging
import difflib
from dataclasses import dataclass, field

from config import settings
from core.retriever import RetrievedChunk
from core.generator import GeneratedAnswer, Citation

logger = logging.getLogger(__name__)


@dataclass
class GroundingDetail:
    """Detailed result of grounding check for a single citation."""
    citation: Citation
    source_text: str          # The chunk text that was cited
    match_ratio: float        # How well the quote matches the source (0-1)
    is_grounded: bool         # Whether the citation is verified

    @property
    def status(self) -> str:
        if self.match_ratio >= 0.8:
            return "strong"
        elif self.match_ratio >= settings.GROUNDING_MATCH_THRESHOLD:
            return "partial"
        else:
            return "ungrounded"


@dataclass
class UnsupportedClaim:
    """A claim in the answer that is not supported by any retrieved chunk."""
    claim: str
    best_match_score: float   # Highest similarity to any chunk
    best_match_chunk_id: str  # Which chunk was closest
    is_supported: bool        # Whether it meets support threshold


@dataclass
class ReliabilityReport:
    """Complete reliability assessment of a generated answer."""
    # Scores (0.0 to 1.0, higher = better)
    citation_score: float = 0.0
    grounding_score: float = 0.0
    confidence: float = 0.0

    # Unsupported claims
    unsupported_claims: list[UnsupportedClaim] = field(default_factory=list)
    unsupported_ratio: float = 0.0  # Fraction of claims that are unsupported

    # Abstention
    should_abstain: bool = False
    abstention_reason: str | None = None

    # Detailed breakdowns
    grounding_details: list[GroundingDetail] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    # Retrieval paths used to answer this question.
    # Subset of {"vector", "graph", "community"}.
    sources_used: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Human-readable verdict."""
        if self.should_abstain:
            return "abstained"
        elif self.confidence >= 0.8:
            return "grounded"
        elif self.confidence >= settings.CONFIDENCE_THRESHOLD:
            return "partially_grounded"
        else:
            return "low_confidence"

    @property
    def verdict_emoji(self) -> str:
        mapping = {
            "grounded": "✅",
            "partially_grounded": "🟡",
            "low_confidence": "🟠",
            "abstained": "🚫",
        }
        return mapping.get(self.verdict, "❓")


class ReliabilityChecker:
    """
    Runs post-generation reliability checks on the answer.
    This is the key engineering contribution of the project.
    """

    def __init__(self):
        self.confidence_weights = settings.CONFIDENCE_WEIGHTS

    def check(
        self,
        answer: GeneratedAnswer,
        chunks: list[RetrievedChunk],
    ) -> ReliabilityReport:
        """
        Run all reliability checks and produce a comprehensive report.

        Args:
            answer: The generated answer with citations.
            chunks: The retrieved context chunks.

        Returns:
            A ReliabilityReport with scores and abstention decision.
        """
        # Build a lookup map: chunk_id → chunk
        chunk_map = {c.chunk_id: c for c in chunks}

        # ── Check 1: Citation Presence ────────────────────────────────────
        citation_score = self._check_citation_presence(answer)

        # ── Check 2: Grounding Verification ───────────────────────────────
        grounding_score, grounding_details = self._check_grounding(
            answer, chunk_map
        )

        # ── Check 3: Unsupported Claim Detection ─────────────────────────
        unsupported_claims = self._check_unsupported_claims(answer, chunks)
        total_claims = len(unsupported_claims)
        unsupported_count = sum(1 for c in unsupported_claims if not c.is_supported)
        unsupported_ratio = unsupported_count / max(total_claims, 1)

        # ── Check 4: Confidence Scoring ───────────────────────────────────
        retrieval_scores = [c.score for c in chunks]
        confidence = self._compute_confidence(
            retrieval_scores=retrieval_scores,
            citation_score=citation_score,
            grounding_score=grounding_score,
            self_confidence=answer.self_confidence,
            unsupported_ratio=unsupported_ratio,
        )

        # ── Check 5: Abstention Decision ──────────────────────────────────
        should_abstain, abstention_reason = self._should_abstain(
            confidence=confidence,
            retrieval_scores=retrieval_scores,
            citation_score=citation_score,
            grounding_score=grounding_score,
            unsupported_ratio=unsupported_ratio,
        )

        sources_used = sorted({c.retrieval_source for c in chunks})

        report = ReliabilityReport(
            citation_score=round(citation_score, 3),
            grounding_score=round(grounding_score, 3),
            confidence=round(confidence, 3),
            unsupported_claims=unsupported_claims,
            unsupported_ratio=round(unsupported_ratio, 3),
            should_abstain=should_abstain,
            abstention_reason=abstention_reason,
            grounding_details=grounding_details,
            details={
                "retrieval_scores": [round(s, 4) for s in retrieval_scores],
                "avg_retrieval_score": round(
                    sum(retrieval_scores) / max(len(retrieval_scores), 1), 4
                ),
                "num_citations": len(answer.citations),
                "self_confidence": answer.self_confidence,
                "confidence_weights": self.confidence_weights,
                "total_claims": total_claims,
                "unsupported_count": unsupported_count,
                "unsupported_ratio": round(unsupported_ratio, 3),
            },
            sources_used=sources_used,
        )

        logger.info(
            f"Reliability: {report.verdict_emoji} {report.verdict} "
            f"(confidence={report.confidence}, citation={report.citation_score}, "
            f"grounding={report.grounding_score}, unsupported={unsupported_ratio:.0%}, "
            f"abstain={report.should_abstain})"
        )

        return report

    # ── Check 1: Citation Presence ────────────────────────────────────────────

    def _check_citation_presence(self, answer: GeneratedAnswer) -> float:
        """
        Check whether the answer includes citations to retrieved sources.

        Scoring:
        - 0.0: No citations at all
        - 0.5: Citations exist but don't reference valid chunk IDs
        - 1.0: Citations exist and reference valid sources
        """
        if not answer.citations:
            logger.debug("Citation check: No citations found in answer")
            return 0.0

        # Check that citations have non-empty quotes
        valid_citations = [
            c for c in answer.citations if c.quote and c.chunk_id
        ]

        if not valid_citations:
            return 0.3  # Has citations structure but empty content

        # Score based on ratio of valid citations
        ratio = len(valid_citations) / len(answer.citations)
        return max(0.3, ratio)  # Floor at 0.3 if any citations exist

    # ── Check 2: Grounding Verification ───────────────────────────────────────

    def _check_grounding(
        self,
        answer: GeneratedAnswer,
        chunk_map: dict[str, RetrievedChunk],
    ) -> tuple[float, list[GroundingDetail]]:
        """
        Verify that cited quotes actually appear in the source chunks.
        Uses fuzzy string matching to account for minor LLM paraphrasing.

        Returns:
            Tuple of (grounding_score, list of GroundingDetail).
        """
        if not answer.citations:
            return 0.0, []

        details = []
        for citation in answer.citations:
            chunk = chunk_map.get(citation.chunk_id)

            if chunk is None:
                # Citation references a chunk that wasn't retrieved
                details.append(
                    GroundingDetail(
                        citation=citation,
                        source_text="[chunk not found]",
                        match_ratio=0.0,
                        is_grounded=False,
                    )
                )
                continue

            if not citation.quote:
                details.append(
                    GroundingDetail(
                        citation=citation,
                        source_text=chunk.text[:100],
                        match_ratio=0.0,
                        is_grounded=False,
                    )
                )
                continue

            # Fuzzy match the cited quote against the source chunk
            match_ratio = self._fuzzy_match(citation.quote, chunk.text)

            details.append(
                GroundingDetail(
                    citation=citation,
                    source_text=chunk.text[:200],
                    match_ratio=round(match_ratio, 3),
                    is_grounded=match_ratio >= settings.GROUNDING_MATCH_THRESHOLD,
                )
            )

        # Overall grounding score = average of all citation match ratios
        if details:
            grounding_score = sum(d.match_ratio for d in details) / len(details)
        else:
            grounding_score = 0.0

        return grounding_score, details

    def _fuzzy_match(self, quote: str, source_text: str) -> float:
        """
        Check how well a quoted string matches within a source text.
        Uses multiple strategies and returns the best match score.
        """
        quote_lower = quote.lower().strip()
        source_lower = source_text.lower().strip()

        # Empty quote cannot match anything
        if not quote_lower:
            return 0.0

        # Strategy 1: Exact substring check
        if quote_lower in source_lower:
            return 1.0

        # Strategy 2: SequenceMatcher on the full texts
        full_ratio = difflib.SequenceMatcher(
            None, quote_lower, source_lower
        ).ratio()

        # Strategy 3: Find best matching subsequence
        # Slide a window of quote length over the source and find best match
        best_window_ratio = 0.0
        quote_len = len(quote_lower)
        if quote_len < len(source_lower):
            # Sample windows to avoid O(n*m) complexity on long texts
            step = max(1, (len(source_lower) - quote_len) // 50)
            for start in range(0, len(source_lower) - quote_len + 1, step):
                window = source_lower[start : start + quote_len]
                ratio = difflib.SequenceMatcher(
                    None, quote_lower, window
                ).ratio()
                best_window_ratio = max(best_window_ratio, ratio)

        return max(full_ratio, best_window_ratio)

    # ── Check 3: Confidence Scoring ───────────────────────────────────────────

    def _compute_confidence(
        self,
        retrieval_scores: list[float],
        citation_score: float,
        grounding_score: float,
        self_confidence: float,
        unsupported_ratio: float = 0.0,
    ) -> float:
        """
        Compute a weighted confidence score from multiple quality signals.

        Weights (from settings):
        - retrieval: 0.30  (are the retrieved chunks relevant?)
        - citation: 0.25   (does the answer cite sources?)
        - grounding: 0.25  (are citations verified?)
        - self_confidence: 0.20 (model's own confidence assessment)

        Penalty: unsupported claims reduce confidence.
        """
        avg_retrieval = (
            sum(retrieval_scores) / len(retrieval_scores)
            if retrieval_scores
            else 0.0
        )

        w = self.confidence_weights
        confidence = (
            w["retrieval"] * avg_retrieval
            + w["citation"] * citation_score
            + w["grounding"] * grounding_score
            + w["self_confidence"] * self_confidence
        )

        # Penalty for unsupported claims (up to 20% reduction)
        confidence *= (1.0 - 0.2 * unsupported_ratio)

        return max(0.0, min(1.0, confidence))

    # ── Check 4: Abstention Decision ──────────────────────────────────────────

    def _check_unsupported_claims(
        self,
        answer: GeneratedAnswer,
        chunks: list[RetrievedChunk],
    ) -> list[UnsupportedClaim]:
        """
        Split the answer into individual claims/sentences and check if each
        is supported by ANY retrieved chunk.

        A claim is considered unsupported if its best fuzzy match against all
        chunks is below the grounding threshold.
        """
        if not answer.answer or not chunks:
            return []

        # Split answer into sentences (claims)
        sentences = self._split_into_claims(answer.answer)

        claims = []
        for sentence in sentences:
            # Skip very short sentences (e.g., "Yes.", citations only)
            if len(sentence.split()) < 4:
                continue

            # Skip sentences that are just citation references
            clean = re.sub(r'\[(?:Source|[VGC]) ?\d+\]', '', sentence).strip()
            if len(clean.split()) < 3:
                continue

            # Check against all chunks
            best_score = 0.0
            best_chunk_id = ""
            for chunk in chunks:
                score = self._fuzzy_match(clean, chunk.text)
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id

            claims.append(UnsupportedClaim(
                claim=sentence,
                best_match_score=round(best_score, 3),
                best_match_chunk_id=best_chunk_id,
                is_supported=best_score >= settings.GROUNDING_MATCH_THRESHOLD,
            ))

        return claims

    def _split_into_claims(self, text: str) -> list[str]:
        """Split text into individual sentences/claims."""
        # Remove markdown formatting
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # bold
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)      # italic
        clean = re.sub(r'\[(?:Source|[VGC]) ?\d+\]', '', clean)  # citations

        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', clean)

        # Also split by newlines (bullet points, etc.)
        result = []
        for s in sentences:
            for line in s.split('\n'):
                line = line.strip().lstrip('- •*')
                if line:
                    result.append(line)

        return result

    def _should_abstain(
        self,
        confidence: float,
        retrieval_scores: list[float],
        citation_score: float,
        grounding_score: float,
        unsupported_ratio: float = 0.0,
    ) -> tuple[bool, str | None]:
        """
        Decide whether the system should abstain from answering.

        Abstains when:
        1. Overall confidence is below threshold
        2. Best retrieval score is too low (no relevant docs found)
        3. No citations AND grounding is zero (complete lack of evidence)
        4. Majority of claims are unsupported
        """
        reasons = []

        # Rule 1: Low overall confidence
        if confidence < settings.CONFIDENCE_THRESHOLD:
            reasons.append(
                f"Overall confidence ({confidence:.2f}) is below threshold "
                f"({settings.CONFIDENCE_THRESHOLD})"
            )

        # Rule 2: No relevant documents retrieved
        best_retrieval = max(retrieval_scores) if retrieval_scores else 0.0
        if best_retrieval < settings.MIN_RETRIEVAL_SCORE:
            reasons.append(
                f"Best retrieval score ({best_retrieval:.2f}) indicates no "
                f"relevant documents were found"
            )

        # Rule 3: No evidence trail at all
        if citation_score == 0.0 and grounding_score == 0.0:
            reasons.append(
                "No citations or grounding evidence found — answer may be "
                "entirely unsupported"
            )

        # Rule 4: Majority of claims unsupported
        if unsupported_ratio > 0.5:
            reasons.append(
                f"{unsupported_ratio:.0%} of answer claims are not supported "
                f"by retrieved documents"
            )

        if reasons:
            return True, "; ".join(reasons)

        return False, None
