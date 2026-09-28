from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Chunk, Page
from app.embeddings.embedding_service import EmbeddingService

@dataclass
class RetrievedChunk:
    content: str
    section: str
    url: str
    page_type: str
    similarity: float  
async def retrieve_relevant_chunks(
    website_id: UUID,
    query: str,
    session: AsyncSession,
    top_k: int = 5,
    embedding_service: EmbeddingService | None = None,
) -> list[RetrievedChunk]:
    embedding_service = embedding_service or EmbeddingService()

    [query_vector] = await embedding_service.embed_texts([query])
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(Chunk, Page.url, Page.page_type, distance.label("distance"))
        .join(Page, Chunk.page_id == Page.id)
        .where(Page.website_id == website_id)
        .where(Chunk.embedding.isnot(None))
        .order_by(distance)
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    return [
        RetrievedChunk(
            content=chunk.content,
            section=chunk.section or "other",
            url=url,
            page_type=page_type,
            similarity=1.0 - float(dist),
        )
        for chunk, url, page_type, dist in rows
    ]
