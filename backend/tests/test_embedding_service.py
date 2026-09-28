import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.embeddings.embedding_service import (
    EmbeddingService, FakeEmbeddingProvider, EmbeddingError,
)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


@pytest.mark.asyncio
async def test_fake_provider_is_deterministic():
    provider = FakeEmbeddingProvider(dimensions=64)
    v1 = await provider.embed_batch(["Starter plan is $19 per month"])
    v2 = await provider.embed_batch(["Starter plan is $19 per month"])
    assert v1 == v2


@pytest.mark.asyncio
async def test_fake_provider_different_text_different_vector():
    provider = FakeEmbeddingProvider(dimensions=64)
    v1, v2 = await provider.embed_batch(["Pricing starts at $19", "Our team was founded in 2020"])
    assert v1 != v2


@pytest.mark.asyncio
async def test_fake_provider_vectors_are_unit_normalized():
    provider = FakeEmbeddingProvider(dimensions=128)
    [v] = await provider.embed_batch(["Some arbitrary text for normalization check"])
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_fake_provider_similar_text_more_similar_than_unrelated():
    """Not a claim of real semantic understanding — just checks the fake
    provider's vectors have SOME structure (shared words -> higher
    cosine similarity), enough to exercise ranking logic meaningfully
    in tests rather than returning arbitrary noise."""
    provider = FakeEmbeddingProvider(dimensions=256)
    texts = [
        "Starter plan pricing is nineteen dollars per month",
        "Pro plan pricing is forty nine dollars per month",
        "Our engineering team was founded in Austin Texas in 2020",
    ]
    v_pricing_a, v_pricing_b, v_about = await provider.embed_batch(texts)

    sim_pricing_pricing = cosine_sim(v_pricing_a, v_pricing_b)
    sim_pricing_about = cosine_sim(v_pricing_a, v_about)
    assert sim_pricing_pricing > sim_pricing_about


@pytest.mark.asyncio
async def test_embedding_service_batches_correctly():
    provider = FakeEmbeddingProvider(dimensions=32)
    service = EmbeddingService(provider=provider, batch_size=3)
    texts = [f"chunk number {i}" for i in range(10)]
    vectors = await service.embed_texts(texts)
    assert len(vectors) == 10
    assert all(len(v) == 32 for v in vectors)


@pytest.mark.asyncio
async def test_embedding_service_empty_input():
    service = EmbeddingService(provider=FakeEmbeddingProvider(dimensions=32))
    assert await service.embed_texts([]) == []


@pytest.mark.asyncio
async def test_openai_provider_requires_api_key():
    from app.embeddings.embedding_service import OpenAIEmbeddingProvider
    with pytest.raises(EmbeddingError):
        OpenAIEmbeddingProvider(api_key="", model="text-embedding-3-small", dimensions=1536)


def test_default_provider_falls_back_to_fake_without_api_key(monkeypatch):
    from app.embeddings.embedding_service import EmbeddingService
    import app.config as config_module

    config_module.get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")  # requested openai...
    service = EmbeddingService()  # ...but no key, so should still fall back
    assert isinstance(service.provider, FakeEmbeddingProvider)
    config_module.get_settings.cache_clear()


# ---- Local (free, key-less) provider ---------------------------------
# The real model is downloaded from the internet on first use, so these
# tests swap in a tiny stand-in model. They verify everything around it:
# padding to the DB column size, cosine-similarity preservation, and
# provider selection.

class _StubLocalModel:
    """Mimics fastembed.TextEmbedding.embed(): yields one 4-dim vector per text."""
    def embed(self, texts):
        for t in texts:
            base = [float(len(t) % 7 + 1), 2.0, 3.0, float(len(t.split()) + 1)]
            norm = math.sqrt(sum(x * x for x in base))
            yield [x / norm for x in base]


def _patch_local_loader(monkeypatch):
    import app.embeddings.embedding_service as es
    monkeypatch.setattr(es, "_load_local_model", lambda name: _StubLocalModel())
    return es


@pytest.mark.asyncio
async def test_local_provider_pads_to_database_dimensions(monkeypatch):
    es = _patch_local_loader(monkeypatch)
    provider = es.LocalEmbeddingProvider(model_name="stub", dimensions=1536)
    vectors = await provider.embed_batch(["hello world", "pricing plans"])
    assert len(vectors) == 2
    assert all(len(v) == 1536 for v in vectors)
    assert all(x == 0.0 for x in vectors[0][4:])  # padding is zeros


@pytest.mark.asyncio
async def test_local_provider_padding_preserves_cosine_similarity(monkeypatch):
    es = _patch_local_loader(monkeypatch)
    small = es.LocalEmbeddingProvider(model_name="stub", dimensions=4)
    padded = es.LocalEmbeddingProvider(model_name="stub", dimensions=1536)
    texts = ["alpha beta", "gamma delta epsilon"]
    a4, b4 = await small.embed_batch(texts)
    a_p, b_p = await padded.embed_batch(texts)
    assert cosine_sim(a_p, b_p) == pytest.approx(cosine_sim(a4, b4), abs=1e-9)


@pytest.mark.asyncio
async def test_local_provider_rejects_model_larger_than_column(monkeypatch):
    es = _patch_local_loader(monkeypatch)
    provider = es.LocalEmbeddingProvider(model_name="stub", dimensions=2)
    with pytest.raises(EmbeddingError):
        await provider.embed_batch(["too big for a 2-dim column"])


def test_provider_setting_local_selects_local_provider(monkeypatch):
    es = _patch_local_loader(monkeypatch)
    import app.config as config_module
    config_module.get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    service = es.EmbeddingService()
    assert isinstance(service.provider, es.LocalEmbeddingProvider)
    config_module.get_settings.cache_clear()
