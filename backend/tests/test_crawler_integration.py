import http.server
import os
import socketserver
import sys
import tempfile
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.crawler.crawler import WebsiteCrawler  # noqa: E402

PAGES = {
    "index.html": """<html><head><title>Acme Corp - Home</title></head>
<body><nav><a href="/about.html">About</a><a href="/pricing.html">Pricing</a>
<a href="/blog/post1.html">Blog</a></nav>
<main><h1>Welcome to Acme</h1><p>We build productivity tools.</p>
<a href="/product.html">Our Product</a></main></body></html>""",
    "about.html": """<html><head><title>About Acme</title></head>
<body><main><h1>About Us</h1><p>Founded in 2020, HQ in Austin.</p></main></body></html>""",
    "pricing.html": """<html><head><title>Pricing</title></head>
<body><main><h1>Pricing</h1><h2>Starter</h2><p>$19/month</p></main></body></html>""",
    "product.html": """<html><head><title>Product</title></head>
<body><main><h1>Acme Flow</h1><p>Automates workflows.</p></main></body></html>""",
    "blog/post1.html": """<html><head><title>Blog Post</title></head>
<body><main><h1>5 Tips</h1><p>Some blog content.</p></main></body></html>""",
    "robots.txt": "User-agent: *\nDisallow: /private\n",
}
PRIVATE_PAGE = {"private/secret.html": "<html><body>Private</body></html>"}

@pytest.fixture(scope="module")
def test_site():
    tmp_dir = tempfile.mkdtemp()
    for rel_path, content in {**PAGES, **PRIVATE_PAGE}.items():
        full_path = os.path.join(tmp_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(
        *a, directory=tmp_dir, **kw
    )
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()
    httpd.server_close()


@pytest.mark.asyncio
async def test_crawler_discovers_nav_only_linked_page(test_site):
    crawler = WebsiteCrawler(
        start_url=f"{test_site}/index.html",
        max_pages=20, max_depth=5, crawl_timeout=10, concurrent_requests=3,
    )
    result = await crawler.crawl()
    urls = [p.url for p in result.pages]
    assert any("blog/post1.html" in u for u in urls), f"blog post missing from {urls}"

@pytest.mark.asyncio
async def test_crawler_respects_robots_disallow(test_site):
    crawler = WebsiteCrawler(
        start_url=f"{test_site}/index.html",
        max_pages=20, max_depth=5, crawl_timeout=10, concurrent_requests=3,
        respect_robots_txt=True,
    )
    result = await crawler.crawl()
    urls = [p.url for p in result.pages]
    assert not any("private" in u for u in urls)

@pytest.mark.asyncio
async def test_crawler_enforces_max_pages(test_site):
    crawler = WebsiteCrawler(
        start_url=f"{test_site}/index.html",
        max_pages=2, max_depth=5, crawl_timeout=10, concurrent_requests=3,
    )
    result = await crawler.crawl()
    assert result.pages_crawled <= 2
    assert result.stopped_reason == "max_pages"

@pytest.mark.asyncio
async def test_crawler_classifies_pages_correctly(test_site):
    crawler = WebsiteCrawler(
        start_url=f"{test_site}/index.html",
        max_pages=20, max_depth=5, crawl_timeout=10, concurrent_requests=3,
    )
    result = await crawler.crawl()
    by_url = {p.url: p.page_type for p in result.pages}
    home = next(v for k, v in by_url.items() if k.endswith("index.html"))
    pricing = next(v for k, v in by_url.items() if k.endswith("pricing.html"))
    assert home == "home"
    assert pricing == "pricing"

@pytest.mark.asyncio
async def test_crawler_no_duplicate_pages(test_site):
    crawler = WebsiteCrawler(
        start_url=f"{test_site}/index.html",
        max_pages=20, max_depth=5, crawl_timeout=10, concurrent_requests=3,
    )
    result = await crawler.crawl()
    urls = [p.url for p in result.pages]
    assert len(urls) == len(set(urls))
