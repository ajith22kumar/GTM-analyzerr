from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import math
from abc import ABC, abstractmethod

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails after retries."""


class EmbeddingProvider(ABC):
    dimensions: int

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""


class FakeEmbeddingProvider(EmbeddingProvider):

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        words = text.lower().split()
        if not words:
            words = [""]
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            for i in range(self.dimensions):
                byte = digest[i % len(digest)]
                # Spread bytes across dims with a per-index salt so
                # different dims of the same word's hash don't collide.
                salted = (byte + i * 31) % 256
                vec[i] += (salted / 255.0) - 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@functools.lru_cache(maxsize=2)
def _load_local_model(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingError(
            "EMBEDDING_PROVIDER=local needs the 'fastembed' package. "
            "Run: pip install -r requirements.txt"
        ) from exc
    return TextEmbedding(model_name=model_name)


class LocalEmbeddingProvider(EmbeddingProvider):

    def __init__(self, model_name: str, dimensions: int):
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = _load_local_model(model_name)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for raw in self._model.embed(texts):
            vec = [float(x) for x in raw]
            if len(vec) > self.dimensions:
                raise EmbeddingError(
                    f"Local model '{self.model_name}' produces {len(vec)}-dim "
                    f"vectors, larger than EMBEDDING_DIMENSIONS={self.dimensions}."
                )
            # Zero-padding keeps cosine similarity exactly the same.
            vec.extend([0.0] * (self.dimensions - len(vec)))
            vectors.append(vec)
        return vectors

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # The model is CPU-bound and synchronous; run it off the event loop
        # so the API stays responsive while indexing.
        return await asyncio.to_thread(self._embed_sync, texts)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str, dimensions: int):
        if not api_key:
            raise EmbeddingError(
                "embedding_provider is 'openai' but no OPENAI_API_KEY is "
                "configured. Set OPENAI_API_KEY in .env, or set "
                "EMBEDDING_PROVIDER=fake for offline testing."
            )
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": texts, "model": self.model},
                timeout=30,
            )
        if resp.status_code != 200:
            raise EmbeddingError(
                f"OpenAI embeddings request failed: {resp.status_code} {resp.text[:500]}"
            )
        data = resp.json()
        
        items = sorted(data["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in items]
        for v in vectors:
            if len(v) != self.dimensions:
                raise EmbeddingError(
                    f"Expected {self.dimensions}-dim embeddings, got {len(v)}. "
                    f"Check EMBEDDING_DIMENSIONS matches EMBEDDING_MODEL."
                )
        return vectors


class EmbeddingService:
    """Batches texts and delegates to the configured provider."""

    def __init__(self, provider: EmbeddingProvider | None = None, batch_size: int | None = None):
        settings = get_settings()
        if provider is None:
            provider = self._build_default_provider(settings)
        self.provider = provider
        self.batch_size = batch_size or settings.embedding_batch_size

    @staticmethod
    def _build_default_provider(settings) -> EmbeddingProvider:
        if settings.embedding_provider == "local":
            return LocalEmbeddingProvider(
                model_name=settings.local_embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        if settings.embedding_provider == "fake" or not settings.openai_api_key:
            if settings.embedding_provider != "fake":
                logger.warning(
                    "No OPENAI_API_KEY configured — falling back to the fake "
                    "embedding provider. Real GTM analysis needs a real "
                    "embedding provider; set OPENAI_API_KEY in .env."
                )
            return FakeEmbeddingProvider(dimensions=settings.embedding_dimensions)
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors = await self.provider.embed_batch(batch)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"Provider returned {len(vectors)} vectors for {len(batch)} inputs"
                )
            results.extend(vectors)
        return results
