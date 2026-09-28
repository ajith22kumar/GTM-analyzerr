from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse, parse_qsl, urlencode

# Query params that don't affect page content and only create duplicate URLs.
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "mc_cid", "mc_eid", "ref", "referrer",
    "_ga", "_gl", "session", "sessionid", "sid", "PHPSESSID",
}

NON_HTML_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".zip", ".rar", ".tar", ".gz", ".mp4", ".mp3", ".avi",
    ".mov", ".woff", ".woff2", ".ttf", ".eot", ".xml", ".json", ".csv",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    if not url:
        return None

    if base_url:
        url = urljoin(base_url, url)

    url, _fragment = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept_params = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    kept_params.sort()
    query = urlencode(kept_params)

    return urlunparse((scheme, netloc, path, "", query, ""))


def is_same_domain(url: str, root_domain: str) -> bool:
    host = urlparse(url).netloc.lower()
    host = host.split(":")[0]  # drop port
    root = root_domain.lower().split(":")[0]
    strip_www = lambda h: h[4:] if h.startswith("www.") else h
    return strip_www(host) == strip_www(root)


def looks_like_non_html(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in NON_HTML_EXTENSIONS)


def path_priority_score(url: str, hints: tuple[str, ...]) -> int:
    path = urlparse(url).path.lower()
    for i, hint in enumerate(hints):
        if hint in path:
            return i
    if path in ("", "/"):
        return -1 
    return len(hints)  

@dataclass
class CrawlFrontier:
    root_domain: str
    priority_hints: tuple[str, ...] = ()
    max_depth: int = 10

    _queued: list[tuple[int, int, str]] = field(default_factory=list)  # (priority, depth, url)
    _seen: set[str] = field(default_factory=set)
    _counter: int = 0

    def add(self, url: str, depth: int) -> bool:
        normalized = normalize_url(url)
        if normalized is None:
            return False
        if normalized in self._seen:
            return False
        if depth > self.max_depth:
            return False
        if not is_same_domain(normalized, self.root_domain):
            return False
        if looks_like_non_html(normalized):
            return False

        self._seen.add(normalized)
        priority = path_priority_score(normalized, self.priority_hints)
        self._queued.append((priority, depth, normalized))
        return True

    def pop_next(self) -> tuple[str, int] | None:
        if not self._queued:
            return None
        self._queued.sort(key=lambda item: (item[0], item[1]))
        priority, depth, url = self._queued.pop(0)
        return url, depth

    def __len__(self) -> int:
        return len(self._queued)

    @property
    def seen_count(self) -> int:
        return len(self._seen)
