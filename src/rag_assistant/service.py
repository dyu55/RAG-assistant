from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from .config import Settings
from .models import Answer, Draft, Evidence, Question
from .providers import ModelClient, ProviderError, extractive_draft
from .retrieval import retrieve
from .store import Store
from .text import normalize, split_document, tokens


def verify(draft: Draft, evidence: list[Evidence]) -> tuple[bool, str]:
    if not draft.claims:
        return False, "The retrieved passages do not provide an answer."
    sources = {item.source_id: item for item in evidence}
    for claim in draft.claims:
        source = sources.get(claim.source_id)
        if not source:
            return False, "The model cited a source that was not retrieved."
        if normalize(claim.quote) not in normalize(source.text):
            return False, "A quoted passage could not be found in its cited source."
        words = set(tokens(claim.text))
        if len(words & set(tokens(claim.quote))) / max(1, len(words)) < 0.45:
            return False, "A claim has insufficient textual support in its quotation."
    return True, "All quoted spans were located in their cited passages."


class KnowledgeService:
    def __init__(self, settings: Settings, client: ModelClient | None = None):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.client = client or ModelClient(settings)
        self.cache: OrderedDict[str, tuple[float, Answer]] = OrderedDict()
        self.lock = threading.RLock()

    def ingest(self, filename: str, content: bytes) -> dict:
        filename = Path(filename.replace("\\", "/")).name.strip()
        if not filename or len(filename) > 200 or any(ord(ch) < 32 for ch in filename):
            raise ValueError("Choose a valid filename of at most 200 characters")
        if len(content) > self.settings.max_upload_bytes:
            raise ValueError("Document exceeds the upload size limit")
        indexing = f"extractor-v1:{self.settings.chunk_size}:{self.settings.chunk_overlap}"
        digest = hashlib.sha256(content + b"\0" + indexing.encode()).hexdigest()
        old = self.store.document(filename)
        if old and old["digest"] == digest:
            _, identity, _ = self.store.snapshot([old["id"]])
            if identity != self.settings.embedding_identity:
                raise ValueError("Embedding model changed; use a new data directory")
            return {**old, "unchanged": True}
        chunks = split_document(
            filename, content, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks or len(chunks) > self.settings.max_chunks:
            raise ValueError("Document has no chunks or exceeds the configured index limit")
        vectors = self.client.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.vector = vector
        return self.store.replace(
            chunks, digest, len(content), self.settings.embedding_identity, self.settings.max_chunks
        )

    def ask(self, question: Question) -> Answer:
        started = time.perf_counter()
        revision, identity, chunks = self.store.snapshot(question.document_ids)
        key = json.dumps(
            [
                revision,
                question.model_dump(),
                self.settings.provider,
                self.settings.model,
                self.settings.embedding_identity,
            ],
            sort_keys=True,
        )
        with self.lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < self.settings.cache_ttl:
                answer = cached[1].model_copy(deep=True)
                answer.id, answer.cached = uuid.uuid4().hex, True
                answer.timings_ms = {"total": round((time.perf_counter() - started) * 1000, 2)}
                self.cache.move_to_end(key)
                self.store.record(answer)
                return answer
        warnings, vector = [], None
        effective_mode = question.mode
        if effective_mode == "auto":
            from .router import route_query

            decision = route_query(question.text)
            effective_mode = decision.mode
            warnings.append(
                f"Adaptive routing selected '{effective_mode}' mode: {decision.reasoning}"
            )

        effective_question = (
            question
            if question.mode == effective_mode
            else question.model_copy(update={"mode": effective_mode})
        )

        if chunks and effective_mode in {"hybrid", "vector"}:
            if identity != self.settings.embedding_identity:
                raise ValueError(
                    "Embedding model differs from the index; select the original model or reimport"
                )
            try:
                vector = self.client.embed([question.text])[0]
            except ProviderError:
                if effective_mode == "vector":
                    raise
                warnings.append("Embedding service unavailable; using keyword and graph retrieval.")
        evidence = retrieve(effective_question, chunks, vector)
        retrieved = time.perf_counter()
        provider = self.settings.provider
        if not evidence:
            draft = Draft(claims=[])
        elif provider == "offline":
            draft = extractive_draft(question.text, evidence)
        else:
            try:
                draft = self.client.generate(question.text, evidence)
            except ProviderError:
                draft = extractive_draft(question.text, evidence)
                provider = "extractive-fallback"
                warnings.append(
                    "Model service unavailable or returned invalid JSON; showing source excerpts."
                )
        generated = time.perf_counter()
        supported, reason = verify(draft, evidence)
        claims = draft.claims if supported else []
        answer = Answer(
            id=uuid.uuid4().hex,
            question=question.text,
            answer="\n\n".join(f"{claim.text} [{claim.source_id}]" for claim in claims)
            if supported
            else "I could not find enough verified evidence to answer this question.",
            claims=claims,
            evidence=evidence,
            status="supported" if supported else "abstained",
            reason=reason,
            provider=provider,
            retrieval_mode=effective_mode,
            confidence=round(sum(e.relevance for e in evidence) / len(evidence), 3)
            if supported
            else 0,
            warnings=warnings,
            revision=revision,
            timings_ms={
                "retrieval": round((retrieved - started) * 1000, 2),
                "generation": round((generated - retrieved) * 1000, 2),
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        with self.lock:
            self.cache[key] = (time.monotonic(), answer.model_copy(deep=True))
            self.cache.move_to_end(key)
            while len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.store.record(answer)
        return answer
