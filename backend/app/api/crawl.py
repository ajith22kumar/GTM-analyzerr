from __future__ import annotations

import logging
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    ChatRequest, ChatResponse, CrawlRequest, PageListResponse, PageSummary,
    RetrievedChunkResponse, WebsiteStatusResponse,
)
from app.config import get_settings
from app.crawler.crawler import WebsiteCrawler
from app.database.database import get_db, session_scope
from app.database.models import Page, Website
from app.rag.pipeline import index_website
from app.rag.retriever import retrieve_relevant_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/websites", tags=["crawl"])
settings = get_settings()


@router.post("", response_model=WebsiteStatusResponse, status_code=202)
async def start_crawl(
    payload: CrawlRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    url = str(payload.url)
    domain = urlparse(url).netloc

    existing = await db.execute(select(Website).where(Website.url == url))
    website = existing.scalar_one_or_none()

    if website is not None and website.status in ("pending", "crawling", "analyzing"):
        raise HTTPException(
            status_code=409,
            detail=f"A crawl for this URL is already {website.status}.",
        )

    if website is None:
        website = Website(domain=domain, url=url, status="pending")
        db.add(website)
        await db.commit()
        await db.refresh(website)
    else:
        await db.execute(delete(Page).where(Page.website_id == website.id))
        website.status = "pending"
        website.pages_discovered = 0
        website.pages_crawled = 0
        website.pages_failed = 0
        website.error_message = None
        await db.commit()
        await db.refresh(website)

    background_tasks.add_task(
        run_crawl_job,
        website_id=website.id,
        url=url,
        max_pages=payload.max_pages or settings.max_pages,
        max_depth=payload.max_depth or settings.max_depth,
    )

    return website


@router.get("/{website_id}", response_model=WebsiteStatusResponse)
async def get_website_status(website_id: UUID, db: AsyncSession = Depends(get_db)):
    website = await db.get(Website, website_id)
    if website is None:
        raise HTTPException(status_code=404, detail="Website not found")
    return website


@router.get("/{website_id}/pages", response_model=PageListResponse)
async def list_pages(website_id: UUID, db: AsyncSession = Depends(get_db)):
    website = await db.get(Website, website_id)
    if website is None:
        raise HTTPException(status_code=404, detail="Website not found")

    result = await db.execute(select(Page).where(Page.website_id == website_id))
    pages = result.scalars().all()
    return PageListResponse(
        website_id=website_id,
        total=len(pages),
        pages=[PageSummary.model_validate(p) for p in pages],
    )


@router.post("/{website_id}/chat", response_model=ChatResponse)
async def chat_with_website(
    website_id: UUID, payload: ChatRequest, db: AsyncSession = Depends(get_db),
):
    website = await db.get(Website, website_id)
    if website is None:
        raise HTTPException(status_code=404, detail="Website not found")
    if website.status not in ("indexed",):
        raise HTTPException(
            status_code=409,
            detail=f"Website is not indexed yet (status: {website.status}). Try again once crawling/indexing finishes.",
        )

    chunks = await retrieve_relevant_chunks(
        website_id, payload.question, db, top_k=payload.top_k or 5,
    )
    return ChatResponse(
        question=payload.question,
        results=[
            RetrievedChunkResponse(
                content=c.content, section=c.section, url=c.url,
                page_type=c.page_type, similarity=c.similarity,
            )
            for c in chunks
        ],
    )

async def run_crawl_job(website_id: UUID, url: str, max_pages: int, max_depth: int) -> None:
    async def update_progress(event: dict) -> None:
        if event.get("event") != "page_crawled" and event.get("event") != "sitemap_checked":
            return
        async with session_scope() as session:
            website = await session.get(Website, website_id)
            if website is None:
                return
            if event["event"] == "sitemap_checked":
                website.sitemap_found = 1 if event.get("found") else 0
            elif event["event"] == "page_crawled":
                website.pages_crawled = event.get("pages_crawled", website.pages_crawled)
                website.pages_discovered = event.get("pages_discovered", website.pages_discovered)
            website.status = "crawling"

    try:
        async with session_scope() as session:
            website = await session.get(Website, website_id)
            website.status = "crawling"

        crawler = WebsiteCrawler(
            start_url=url,
            max_pages=max_pages,
            max_depth=max_depth,
            crawl_timeout=settings.crawl_timeout,
            concurrent_requests=settings.concurrent_requests,
            retry_attempts=settings.retry_attempts,
            respect_robots_txt=settings.respect_robots_txt,
            user_agent=settings.user_agent,
            priority_hints=settings.priority_path_hints,
            progress_callback=update_progress,
        )
        result = await crawler.crawl()

        async with session_scope() as session:
            website = await session.get(Website, website_id)
            for crawled in result.pages:
                page = Page(
                    website_id=website_id,
                    url=crawled.url,
                    canonical_url=crawled.canonical_url,
                    title=crawled.title,
                    meta_description=crawled.meta_description,
                    h1=crawled.h1,
                    headings=crawled.headings,
                    main_content=crawled.main_content,
                    content_blocks=crawled.content_blocks,
                    page_type=crawled.page_type,
                    language=crawled.language,
                    status_code=crawled.status_code,
                    depth=crawled.depth,
                    crawl_status="ok" if crawled.error is None else crawled.error,
                    error=crawled.error,
                )
                session.add(page)

            website.pages_discovered = result.pages_discovered
            website.pages_crawled = result.pages_crawled
            website.pages_failed = result.pages_failed
            website.sitemap_found = 1 if result.sitemap_found else 0
            website.status = "crawled"
        try:
            async with session_scope() as session:
                await index_website(website_id, session)
                website = await session.get(Website, website_id)
                website.status = "indexed"
        except Exception:
            logger.exception("Indexing failed for website_id=%s (crawl itself succeeded)", website_id)
            async with session_scope() as session:
                website = await session.get(Website, website_id)
                if website is not None:
                    website.status = "crawled_index_failed"

    except Exception as exc:  # noqa: BLE001 - top-level job guard, must not raise
        logger.exception("Crawl job failed for website_id=%s", website_id)
        async with session_scope() as session:
            website = await session.get(Website, website_id)
            if website is not None:
                website.status = "failed"
                website.error_message = str(exc)[:2000]
