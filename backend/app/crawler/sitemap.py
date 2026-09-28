from __future__ import annotations

import gzip
import logging
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
MAX_NESTED_SITEMAPS = 20  # safety cap against malicious/huge sitemap indexes


async def discover_sitemap_urls(base_url: str, client: httpx.AsyncClient) -> list[str]:
    parsed = urlparse(base_url)
    domain = f"{parsed.scheme}://{parsed.netloc}"

    candidate_sitemaps = [f"{domain}/sitemap.xml", f"{domain}/sitemap_index.xml"]

    try:
        resp = await client.get(f"{domain}/robots.txt", timeout=10, follow_redirects=True)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.strip().lower().startswith("sitemap:"):
                    sm_url = line.split(":", 1)[1].strip()
                    if sm_url and sm_url not in candidate_sitemaps:
                        candidate_sitemaps.insert(0, sm_url)
    except (httpx.HTTPError, httpx.TimeoutException):
        pass

    all_urls: list[str] = []
    seen_sitemaps: set[str] = set()

    for sm_url in candidate_sitemaps:
        urls = await _fetch_and_parse_sitemap(sm_url, client, seen_sitemaps)
        all_urls.extend(urls)
        if urls:
            break 
    deduped = list(dict.fromkeys(all_urls))
    return deduped

async def _fetch_and_parse_sitemap(
    sitemap_url: str,
    client: httpx.AsyncClient,
    seen: set[str],
    depth: int = 0,
) -> list[str]:
    if sitemap_url in seen or depth > 3 or len(seen) > MAX_NESTED_SITEMAPS:
        return []
    seen.add(sitemap_url)

    try:
        resp = await client.get(sitemap_url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []

        raw_bytes = resp.content
        if sitemap_url.endswith(".gz") or resp.headers.get("content-type", "").endswith("gzip"):
            try:
                raw_bytes = gzip.decompress(raw_bytes)
            except OSError:
                pass  # not actually gzipped; try parsing as-is

        root = ElementTree.fromstring(raw_bytes)
    except (httpx.HTTPError, httpx.TimeoutException, ElementTree.ParseError) as exc:
        logger.info("Could not parse sitemap %s: %s", sitemap_url, exc)
        return []

    tag = root.tag.lower()
    if tag.endswith("sitemapindex"):
        nested_urls: list[str] = []
        for sitemap_el in root.findall(f"{SITEMAP_NS}sitemap"):
            loc_el = sitemap_el.find(f"{SITEMAP_NS}loc")
            if loc_el is not None and loc_el.text:
                nested = await _fetch_and_parse_sitemap(
                    loc_el.text.strip(), client, seen, depth + 1
                )
                nested_urls.extend(nested)
        return nested_urls
    page_urls: list[str] = []
    for url_el in root.findall(f"{SITEMAP_NS}url"):
        loc_el = url_el.find(f"{SITEMAP_NS}loc")
        if loc_el is not None and loc_el.text:
            page_urls.append(urljoin(sitemap_url, loc_el.text.strip()))
    return page_urls
