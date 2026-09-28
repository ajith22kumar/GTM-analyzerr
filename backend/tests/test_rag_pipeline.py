import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database.models import Base, Chunk, Page, Website  # noqa: E402
from app.embeddings.embedding_service import EmbeddingService, FakeEmbeddingProvider  # noqa: E402
from app.rag.pipeline import index_website  # noqa: E402
from app.rag.retriever import retrieve_relevant_chunks  # noqa: E402

settings = get_settings()

async def _db_reachable() -> bool:
    try:
        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: None)
        await engine.dispose()
        return True
    except Exception:
        return False

@pytest.fixture(scope="module")
async def db_session():
    if not await _db_reachable():
        pytest.skip("Postgres not reachable at DATABASE_URL — skipping RAG integration tests")

    from app.database.database import AsyncSessionLocal, engine

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        yield session

def _content_blocks_for(sections: dict[str, str]) -> list[dict]:
    blocks = []
    for heading, text_ in sections.items():
        blocks.append({"tag": "h2", "text": heading})
        blocks.append({"tag": "p", "text": text_})
    return blocks

@pytest.fixture
def fake_embedder():
    return EmbeddingService(provider=FakeEmbeddingProvider(dimensions=settings.embedding_dimensions))

@pytest.mark.asyncio
async def test_index_website_creates_expected_chunk_count(db_session, fake_embedder):
    domain = f"test-{uuid.uuid4().hex[:8]}.example.com"
    website = Website(domain=domain, url=f"https://{domain}", status="crawled")
    db_session.add(website)
    await db_session.commit()
    await db_session.refresh(website)

    page = Page(
        website_id=website.id, url=f"https://{domain}/product", title="Product",
        page_type="product", depth=1, status_code=200,
        content_blocks=_content_blocks_for({
            "Overview": "This product automates repetitive workflows for remote teams every single day.",
            "Pricing": "Starter plan is nineteen dollars per month, billed annually with no setup fee at all.",
        }),
    )
    db_session.add(page)
    await db_session.commit()

    summary = await index_website(website.id, db_session, embedding_service=fake_embedder)
    await db_session.commit()

    assert summary["pages_indexed"] == 1
    assert summary["chunks_created"] == 2  # Overview section + Pricing section

    result = await db_session.execute(select(Chunk).where(Chunk.page_id == page.id))
    chunks = result.scalars().all()
    assert len(chunks) == 2
    assert all(c.embedding is not None for c in chunks)
    assert all(len(c.embedding) == settings.embedding_dimensions for c in chunks)

@pytest.mark.asyncio
async def test_reindexing_is_idempotent_not_duplicative(db_session, fake_embedder):
    domain = f"test-{uuid.uuid4().hex[:8]}.example.com"
    website = Website(domain=domain, url=f"https://{domain}", status="crawled")
    db_session.add(website)
    await db_session.commit()
    await db_session.refresh(website)

    page = Page(
        website_id=website.id, url=f"https://{domain}/about", title="About",
        page_type="about", depth=1, status_code=200,
        content_blocks=_content_blocks_for({
            "Our Story": "Founded in twenty twenty in Austin Texas by a small team of engineers.",
        }),
    )
    db_session.add(page)
    await db_session.commit()

    await index_website(website.id, db_session, embedding_service=fake_embedder)
    await db_session.commit()
    await index_website(website.id, db_session, embedding_service=fake_embedder)  # re-run
    await db_session.commit()

    result = await db_session.execute(select(Chunk).where(Chunk.page_id == page.id))
    chunks = result.scalars().all()
    assert len(chunks) == 1  # not 2 — old chunks were cleared before re-indexing


@pytest.mark.asyncio
async def test_retrieval_ranks_most_relevant_chunk_first(db_session):
    domain = f"test-{uuid.uuid4().hex[:8]}.example.com"
    website = Website(domain=domain, url=f"https://{domain}", status="crawled")
    db_session.add(website)
    await db_session.commit()
    await db_session.refresh(website)

    page = Page(
        website_id=website.id, url=f"https://{domain}/pricing", title="Pricing",
        page_type="pricing", depth=1, status_code=200,
    )
    db_session.add(page)
    await db_session.commit()
    await db_session.refresh(page)

    dims = settings.embedding_dimensions

    def basis_vector(index: int) -> list[float]:
        v = [0.0] * dims
        v[index] = 1.0
        return v
    pricing_vec = basis_vector(0)
    about_vec = basis_vector(1)
    faq_vec = basis_vector(2)

    db_session.add_all([
        Chunk(page_id=page.id, chunk_index=0, content="Pricing chunk content", section="pricing", embedding=pricing_vec),
        Chunk(page_id=page.id, chunk_index=1, content="About chunk content", section="about", embedding=about_vec),
        Chunk(page_id=page.id, chunk_index=2, content="FAQ chunk content", section="faq", embedding=faq_vec),
    ])
    await db_session.commit()

    query_vec = basis_vector(0)
    query_vec[5] = 0.01

    class _FixedVectorProvider:
        dimensions = dims
        async def embed_batch(self, texts):
            return [query_vec for _ in texts]

    fixed_embedder = EmbeddingService(provider=_FixedVectorProvider())

    results = await retrieve_relevant_chunks(
        website.id, "what does pricing cost", db_session, top_k=3,
        embedding_service=fixed_embedder,
    )

    assert len(results) == 3
    assert results[0].section == "pricing"
    assert results[0].similarity > results[1].similarity
    assert results[0].similarity > results[2].similarity
    assert results[0].similarity == pytest.approx(1.0, abs=1e-3)
    assert results[0].url == f"https://{domain}/pricing"


@pytest.mark.asyncio
async def test_retrieval_scoped_to_single_website(db_session, fake_embedder):
    domain_a = f"test-{uuid.uuid4().hex[:8]}.example.com"
    domain_b = f"test-{uuid.uuid4().hex[:8]}.example.com"

    website_a = Website(domain=domain_a, url=f"https://{domain_a}", status="crawled")
    website_b = Website(domain=domain_b, url=f"https://{domain_b}", status="crawled")
    db_session.add_all([website_a, website_b])
    await db_session.commit()
    await db_session.refresh(website_a)
    await db_session.refresh(website_b)

    page_a = Page(website_id=website_a.id, url=f"https://{domain_a}/x", page_type="other", depth=0, status_code=200,
                  content_blocks=_content_blocks_for({"Info": "Website A specific unique content about widgets."}))
    page_b = Page(website_id=website_b.id, url=f"https://{domain_b}/x", page_type="other", depth=0, status_code=200,
                  content_blocks=_content_blocks_for({"Info": "Website B specific unique content about gadgets."}))
    db_session.add_all([page_a, page_b])
    await db_session.commit()

    await index_website(website_a.id, db_session, embedding_service=fake_embedder)
    await index_website(website_b.id, db_session, embedding_service=fake_embedder)
    await db_session.commit()

    results = await retrieve_relevant_chunks(
        website_a.id, "widgets", db_session, top_k=10, embedding_service=fake_embedder,
    )
    assert len(results) > 0
    assert all(r.url.startswith(f"https://{domain_a}") for r in results)
