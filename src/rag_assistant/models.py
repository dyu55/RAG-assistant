from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Chunk(BaseModel):
    id: str
    document_id: str
    filename: str
    text: str
    page: int
    start: int
    end: int
    vector: list[float] = Field(default_factory=list, exclude=True)
    entities: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page: int
    start: int
    end: int
    relevance: float
    fusion_score: float
    channels: list[str]
    path: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    source_id: str = Field(min_length=1, max_length=30)
    quote: str = Field(min_length=6, max_length=4000)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(max_length=8)


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=12)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    mode: Literal["hybrid", "keyword", "vector", "graph", "auto"] = "hybrid"

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a question")
        return value.strip()


class TriadMetrics(BaseModel):
    context_relevance: float = 0.0
    groundedness: float = 0.0
    answer_relevance: float = 0.0
    composite_score: float = 0.0
    verdict: str = "supported"


class Answer(BaseModel):
    id: str
    question: str
    answer: str
    claims: list[Claim]
    evidence: list[Evidence]
    status: Literal["supported", "abstained"]
    reason: str
    provider: str
    retrieval_mode: str
    confidence: float
    warnings: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    cached: bool = False
    revision: int
    evaluation: TriadMetrics | None = None
