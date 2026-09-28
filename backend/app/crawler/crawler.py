from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

from app.crawler.robots import RobotsChecker
from app.crawler.sitemap import discover_sitemap_urls
from app.crawler.url_manager import CrawlFrontier, normalize_url
from app.extraction.cleaner import clean_structured_blocks, clean_text_blocks, strip_noise_from_html
from app.extraction.html_parser import parse_html
from app.extraction.page_classifier import classify_page

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
ProgressCallback = Optional[Callable[[dict], Awaitable[None]]]


@dataclass
class CrawledPage:
    url: str
    canonical_url: str
    title: str
    meta_description: str
    h1: str
    headings: list[str]
    main_content: str          
    content_blocks: list[dict]
    page_type: str
    language: str
    status_code: int
    depth: int
    crawl_timestamp: str
    error: Optional[str] = None


@dataclass
class CrawlResult:
    website_url: str
    pages: list[CrawledPage] = field(default_factory=list)
    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    sitemap_found: bool = False
    stopped_reason: str = "completed" 


class WebsiteCrawler:
    def __init__(
        self,
        start_url: str,
        max_pages: int = 500,
        max_depth: int = 10,
        crawl_timeout: int = 30,
        concurrent_requests: int = 5,
        retry_attempts: int = 2,
        respect_robots_txt: bool = True,
        user_agent: str = "GTM-Analyzer-Bot/1.0",
        priority_hints: tuple[str, ...] = (),
        already_crawled_urls: set[str] | None = None,
        progress_callback: ProgressCallback = None,
    ):
        self.start_url = start_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.crawl_timeout = crawl_timeout
        self.concurrent_requests = concurrent_requests
        self.retry_attempts = retry_attempts
        self.respect_robots_txt = respect_robots_txt
        self.user_agent = user_agent
        self.already_crawled = already_crawled_urls or set()
        self.progress_callback = progress_callback

        root_domain = urlparse(start_url).netloc
        self.frontier = CrawlFrontier(
            root_domain=root_domain, priority_hints=priority_hints, max_depth=max_depth
        )
        self.robots = RobotsChecker(user_agent=user_agent, timeout=min(crawl_timeout, 10))
        self._semaphore = asyncio.Semaphore(concurrent_requests)

    async def _report(self, event: str, **kwargs) -> None:
        if self.progress_callback:
            await self.progress_callback({"event": event, **kwargs})
        logger.info("crawl_progress event=%s %s", event, kwargs)

    async def _fetch_with_retry(self, client: httpx.AsyncClient, url: str) -> httpx.Response | None:
        last_exc: Exception | None = None
        for attempt in range(self.retry_attempts + 1):
            try:
                resp = await client.get(
                    url, timeout=self.crawl_timeout, follow_redirects=True,
                    headers={"User-Agent": self.user_agent},
                )
                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self.retry_attempts:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                return resp
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self.retry_attempts:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
        if last_exc:
            logger.warning("Failed to fetch %s after retries: %s", url, last_exc)
        return None

    async def _process_url(
        self, client: httpx.AsyncClient, url: str, depth: int
    ) -> CrawledPage | None:
        async with self._semaphore:
            if self.respect_robots_txt:
                allowed = await self.robots.is_allowed(url, client)
                if not allowed:
                    await self._report("page_skipped_robots", url=url)
                    return None

            response = await self._fetch_with_retry(client, url)

        if response is None:
            return CrawledPage(
                url=url, canonical_url=url, title="", meta_description="", h1="",
                headings=[], main_content="", content_blocks=[], page_type="other", language="",
                status_code=0, depth=depth,
                crawl_timestamp=datetime.now(timezone.utc).isoformat(),
                error="fetch_failed",
            )

        content_type = response.headers.get("content-type", "")
        if response.status_code >= 400 or "text/html" not in content_type:
            return CrawledPage(
                url=url, canonical_url=str(response.url), title="", meta_description="",
                h1="", headings=[], main_content="", content_blocks=[], page_type="other", language="",
                status_code=response.status_code, depth=depth,
                crawl_timestamp=datetime.now(timezone.utc).isoformat(),
                error=f"http_{response.status_code}" if response.status_code >= 400 else "non_html",
            )
        raw_parsed = parse_html(response.text, url)
        for link in raw_parsed.links:
            normalized = normalize_url(link, base_url=url)
            if normalized:
                self.frontier.add(normalized, depth + 1)

        cleaned_html = strip_noise_from_html(response.text)
        parsed = parse_html(cleaned_html, url)
        text_blocks = clean_text_blocks(parsed.raw_text_blocks)
        structured_blocks = clean_structured_blocks(parsed.structured_blocks)
        page_type = classify_page(url, parsed.title, parsed.headings)

        return CrawledPage(
            url=url,
            canonical_url=str(response.url),
            title=parsed.title,
            meta_description=parsed.meta_description,
            h1=parsed.h1,
            headings=parsed.headings,
            main_content="\n\n".join(text_blocks),
            content_blocks=[{"tag": b.tag, "text": b.text} for b in structured_blocks],
            page_type=page_type,
            language=parsed.language,
            status_code=response.status_code,
            depth=depth,
            crawl_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def crawl(self) -> CrawlResult:
        result = CrawlResult(website_url=self.start_url)

        async with httpx.AsyncClient(http2=False) as client:
            sitemap_urls = await discover_sitemap_urls(self.start_url, client)
            result.sitemap_found = len(sitemap_urls) > 0
            await self._report(
                "sitemap_checked", found=result.sitemap_found, count=len(sitemap_urls)
            )

            self.frontier.add(self.start_url, 0)
            for su in sitemap_urls:
                self.frontier.add(su, 1)

            result.pages_discovered = self.frontier.seen_count
            await self._report("pages_discovered", count=result.pages_discovered)

            in_flight: set[asyncio.Task] = set()

            while True:
               
                if result.pages_crawled >= self.max_pages:
                    result.stopped_reason = "max_pages"
                    break

                while len(in_flight) < self.concurrent_requests and len(self.frontier):
                    if result.pages_crawled + len(in_flight) >= self.max_pages:
                        break
                    next_item = self.frontier.pop_next()
                    if next_item is None:
                        break
                    next_url, depth = next_item
                    if next_url in self.already_crawled:
                        continue
                    task = asyncio.create_task(self._process_url(client, next_url, depth))
                    in_flight.add(task)

                if not in_flight:
                    break  # frontier exhausted, nothing left to crawl

                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    page = task.result()
                    if page is None:
                        continue
                    result.pages.append(page)
                    if page.error:
                        result.pages_failed += 1
                    else:
                        result.pages_crawled += 1
                    result.pages_discovered = self.frontier.seen_count
                    await self._report(
                        "page_crawled",
                        url=page.url,
                        status=page.status_code,
                        ok=page.error is None,
                        pages_crawled=result.pages_crawled,
                        pages_discovered=result.pages_discovered,
                    )

        await self._report(
            "crawl_completed",
            pages_crawled=result.pages_crawled,
            pages_failed=result.pages_failed,
            stopped_reason=result.stopped_reason,
        )
        return result
