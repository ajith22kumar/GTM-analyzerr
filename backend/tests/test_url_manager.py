import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.crawler.url_manager import (
    normalize_url, is_same_domain, looks_like_non_html, CrawlFrontier,
)

def test_normalize_strips_tracking_params_and_sorts():
    result = normalize_url("https://Example.com/Page/?utm_source=x&b=2&a=1")
    assert result == "https://example.com/Page?a=1&b=2"

def test_normalize_strips_fragment():
    assert normalize_url("https://example.com/page#section") == "https://example.com/page"

def test_normalize_resolves_relative_urls():
    assert normalize_url("/about", "https://example.com/products") == "https://example.com/about"

def test_normalize_rejects_non_http_schemes():
    assert normalize_url("mailto:test@example.com") is None
    assert normalize_url("javascript:void(0)") is None
    assert normalize_url("tel:+123456") is None

def test_normalize_root_keeps_trailing_slash():
    assert normalize_url("https://example.com/") == "https://example.com/"

def test_same_domain_ignores_www():
    assert is_same_domain("https://www.example.com/x", "example.com") is True
    assert is_same_domain("https://example.com/x", "www.example.com") is True

def test_same_domain_rejects_other_domains():
    assert is_same_domain("https://other.com/x", "example.com") is False

def test_non_html_extension_detection():
    assert looks_like_non_html("https://example.com/file.pdf") is True
    assert looks_like_non_html("https://example.com/image.jpg") is True
    assert looks_like_non_html("https://example.com/about") is False

def test_frontier_dedupes_and_rejects_offdomain():
    frontier = CrawlFrontier(root_domain="example.com", max_depth=3)
    assert frontier.add("https://example.com/", 0) is True
    assert frontier.add("https://example.com/", 0) is False  # duplicate
    assert frontier.add("https://external.com/page", 1) is False  # off-domain
    assert frontier.add("https://example.com/file.pdf", 1) is False  # non-html
    assert frontier.add("https://example.com/deep", 99) is False  # over max_depth

def test_frontier_prioritizes_by_hint_order():
    frontier = CrawlFrontier(
        root_domain="example.com",
        priority_hints=("pricing", "product", "about"),
        max_depth=3,
    )
    frontier.add("https://example.com/", 0)
    frontier.add("https://example.com/random-post", 1)
    frontier.add("https://example.com/pricing", 1)
    order = []
    while len(frontier):
        url, _ = frontier.pop_next()
        order.append(url)
    assert order == [
        "https://example.com/",
        "https://example.com/pricing",
        "https://example.com/random-post",
    ]
