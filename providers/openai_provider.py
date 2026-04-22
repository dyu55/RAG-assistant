"""
OpenAI LLM Provider.
Handles both text generation (chat completions) and embedding generation.
"""
from __future__ import annotations

import json
import logging
import time
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from config import settings
from providers.base import Provider

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds


class OpenAIProvider(Provider):
    """Wraps the OpenAI API for text generation and embeddings."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in .env or pass it directly."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            timeout=DEFAULT_TIMEOUT,
            max_retries=MAX_RETRIES,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.3,
    ) -> str:
        """Generate a text response from the LLM with retry on transient failures."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"OpenAI request timed out: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Always respond with valid JSON.",
        temperature: float = 0.3,
    ) -> dict:
        """Generate a structured JSON response from the LLM with retry on transient failures."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"OpenAI request timed out: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI JSON generation failed: {e}")
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using OpenAI API."""
        try:
            # OpenAI allows up to 2048 texts per batch
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def get_model_name(self) -> str:
        """Return the current model identifier."""
        return self.model
