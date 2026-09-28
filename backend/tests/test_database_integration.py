import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import delete, select  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database.models import Base, Chunk, Page, Website  # noqa: E402

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
        pytest.skip("Postgres not reachable at DATABASE_URL — skipping DB integration tests")

    from app.database.database import AsyncSessionLocal, engine

    async with engine.begin() as conn:
        from sqlalchemy import text
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        yield session

@pytest.mark.asyncio
async def test_website_page_chunk_roundtrip(db_session):
    domain = f"test-{uuid.uuid4().hex[:8]}.example.com"
    website = Website(domain=domain, url=f"https://{domain}", status="crawled")
    db_session.add(website)
    await db_session.commit()
    await db_session.refresh(website)

    page = Page(
        website_id=website.id, url=f"https://{domain}/pricing",
        title="Pricing", page_type="pricing", main_content="Starter $19/mo",
        depth=1, status_code=200,
    )
    db_session.add(page)
    await db_session.commit()
    await db_session.refresh(page)

    chunk = Chunk(
        page_id=page.id, chunk_index=0, content="Starter plan is $19/mo",
        embedding=[0.01] * settings.embedding_dimensions,
    )
    db_session.add(chunk)
    await db_session.commit()

    result = await db_session.execute(select(Chunk).where(Chunk.page_id == page.id))
    fetched = result.scalar_one()
    assert len(fetched.embedding) == settings.embedding_dimensions
    assert fetched.content == "Starter plan is $19/mo"

    # cascade delete: removing the website should remove its pages+chunks
    await db_session.execute(delete(Website).where(Website.id == website.id))
    await db_session.commit()

    remaining = await db_session.execute(select(Page).where(Page.website_id == website.id))
    assert remaining.scalar_one_or_none() is None

@pytest.mark.asyncio
async def test_website_url_uniqueness_enforced(db_session):
    domain = f"test-{uuid.uuid4().hex[:8]}.example.com"
    url = f"https://{domain}"
    db_session.add(Website(domain=domain, url=url, status="pending"))
    await db_session.commit()

    db_session.add(Website(domain=domain, url=url, status="pending"))
    with pytest.raises(Exception):  # IntegrityError, wrapped by the driver
        await db_session.commit()
    await db_session.rollback()
