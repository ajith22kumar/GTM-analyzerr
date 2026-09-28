from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking.dynamic_chunker import chunk_page
from app.database.models import Chunk, Page, Website
from app.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

async def index_website(website_id: UUID, session: AsyncSession, embedding_service: EmbeddingService | None = None) -> dict:
    embedding_service = embedding_service or EmbeddingService()

    website = await session.get(Website, website_id)
    if website is None:
        raise ValueError(f"Website {website_id} not found")

    result = await session.execute(select(Page).where(Page.website_id == website_id))
    pages = result.scalars().all()

    page_ids = [p.id for p in pages]
    if page_ids:
        await session.execute(delete(Chunk).where(Chunk.page_id.in_(page_ids)))

    all_text_chunks = []  # list[(Page, TextChunk)]
    for page in pages:
        if not page.content_blocks:
            continue
        from app.extraction.html_parser import ContentBlock
        blocks = [ContentBlock(tag=b["tag"], text=b["text"]) for b in page.content_blocks]
        text_chunks = chunk_page(
            page_type=page.page_type,
            structured_blocks=blocks,
            url=page.url,
            extra_metadata={"website_domain": website.domain},
        )
        for tc in text_chunks:
            all_text_chunks.append((page, tc))

    if not all_text_chunks:
        return {"pages_indexed": len(pages), "chunks_created": 0}

    texts = [tc.content for _page, tc in all_text_chunks]
    vectors = await embedding_service.embed_texts(texts)

    for (page, tc), vector in zip(all_text_chunks, vectors):
        chunk = Chunk(
            page_id=page.id,
            chunk_index=tc.chunk_index,
            content=tc.content,
            section=tc.section,
            embedding=vector,
            chunk_metadata=tc.metadata,
        )
        session.add(chunk)

    logger.info(
        "Indexed website_id=%s: %d pages -> %d chunks",
        website_id, len(pages), len(all_text_chunks),
    )
    return {"pages_indexed": len(pages), "chunks_created": len(all_text_chunks)}
