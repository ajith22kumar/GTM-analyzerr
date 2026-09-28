from __future__ import annotations

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)


class RobotsChecker:
    def __init__(self, user_agent: str, timeout: int = 10):
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser] = {}
        self._crawl_delays: dict[str, float | None] = {}
        self._fetch_failed: set[str] = set()

    @staticmethod
    def _domain_key(url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def _ensure_loaded(self, url: str, client: httpx.AsyncClient) -> None:
        domain = self._domain_key(url)
        if domain in self._parsers or domain in self._fetch_failed:
            return

        robots_url = f"{domain}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            resp = await client.get(
                robots_url, timeout=self.timeout, follow_redirects=True
            )
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
                self._parsers[domain] = parser
                try:
                    delay = parser.crawl_delay(self.user_agent)
                except Exception:
                    delay = None
                self._crawl_delays[domain] = delay
            else:
                # No robots.txt (404 etc.) => everything allowed.
                self._fetch_failed.add(domain)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("robots.txt fetch failed for %s: %s", domain, exc)
            self._fetch_failed.add(domain)

    async def is_allowed(self, url: str, client: httpx.AsyncClient) -> bool:
        await self._ensure_loaded(url, client)
        domain = self._domain_key(url)
        parser = self._parsers.get(domain)
        if parser is None:
            return True  # no robots.txt found / unreachable -> allow
        return parser.can_fetch(self.user_agent, url)

    async def get_crawl_delay(self, url: str, client: httpx.AsyncClient) -> float | None:
        await self._ensure_loaded(url, client)
        domain = self._domain_key(url)
        return self._crawl_delays.get(domain)
