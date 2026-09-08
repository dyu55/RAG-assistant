from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Settings(BaseModel):
    data_dir: Path = Path(".rag-assistant")
    provider: Literal["offline", "ollama", "openai"] = "offline"
    model: str = ""
    base_url: str = "http://localhost:11434"
    api_key: str = Field(default="", repr=False)
    embedding_provider: Literal["local", "ollama", "openai"] = "local"
    embedding_model: str = ""
    embedding_url: str = "http://localhost:11434"
    embedding_api_key: str = Field(default="", repr=False)
    chunk_size: int = Field(default=900, ge=120, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, ge=1024)
    max_chunks: int = Field(default=5000, ge=1)
    request_timeout: float = Field(default=60, gt=0, le=300)
    cache_ttl: float = Field(default=300, ge=0)

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("chunk_overlap must be smaller than half of chunk_size")
        if self.provider != "offline" and not self.model:
            raise ValueError("RAG_MODEL is required for a model provider")
        if self.embedding_provider != "local" and not self.embedding_model:
            raise ValueError("RAG_EMBEDDING_MODEL is required for remote embeddings")
        for url in (self.base_url, self.embedding_url):
            if not url.startswith(("http://", "https://")):
                raise ValueError("Provider URLs must use HTTP or HTTPS")
        return self

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            **{
                name: os.environ[f"RAG_{name.upper()}"]
                for name in cls.model_fields
                if f"RAG_{name.upper()}" in os.environ
            }
        )

    @property
    def embedding_identity(self) -> str:
        if self.embedding_provider == "local":
            return "token-hash-v1:384"
        return f"{self.embedding_provider}:{self.embedding_url.rstrip('/')}:{self.embedding_model}"
