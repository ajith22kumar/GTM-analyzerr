import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.extraction.html_parser import parse_html
from app.extraction.cleaner import strip_noise_from_html, clean_text_blocks, dedupe_blocks
from app.extraction.page_classifier import classify_page

SAMPLE_HTML = """
<html lang="en">
<head>
  <title>Pricing Plans - Coirei</title>
  <meta name="description" content="Explore Coirei pricing plans.">
  <script>console.log('tracking')</script>
</head>
<body>
  <nav class="site-nav"><a href="/">Home</a><a href="/pricing">Pricing</a></nav>
  <div id="cookie-banner">We use cookies. <button>Accept</button></div>
  <header><h1>Simple, transparent pricing</h1></header>
  <main>
    <h2>Starter Plan</h2>
    <p>$29/month billed annually.</p>
    <ul><li>Advanced reporting</li><li>Advanced reporting</li></ul>
  </main>
  <footer class="site-footer">
    <p>&copy; 2026 Coirei Inc. All rights reserved.</p>
  </footer>
</body>
</html>
"""


def test_header_h1_survives_noise_stripping():
    """Regression test: <header> must NOT be blanket-stripped, since real
    page H1s commonly live inside it."""
    cleaned = strip_noise_from_html(SAMPLE_HTML)
    parsed = parse_html(cleaned, "https://coirei.com/pricing")
    assert parsed.h1 == "Simple, transparent pricing"


def test_nav_and_cookie_banner_removed():
    cleaned = strip_noise_from_html(SAMPLE_HTML)
    parsed = parse_html(cleaned, "https://coirei.com/pricing")
    blocks = clean_text_blocks(parsed.raw_text_blocks)
    joined = " ".join(blocks)
    assert "cookies" not in joined.lower()
    assert "Home" not in blocks  # nav link text shouldn't leak in


def test_footer_boilerplate_filtered():
    cleaned = strip_noise_from_html(SAMPLE_HTML)
    parsed = parse_html(cleaned, "https://coirei.com/pricing")
    blocks = clean_text_blocks(parsed.raw_text_blocks)
    assert not any("all rights reserved" in b.lower() for b in blocks)


def test_duplicate_list_items_deduped():
    cleaned = strip_noise_from_html(SAMPLE_HTML)
    parsed = parse_html(cleaned, "https://coirei.com/pricing")
    blocks = clean_text_blocks(parsed.raw_text_blocks)
    assert blocks.count("Advanced reporting") == 1


def test_links_extracted_from_raw_html_include_nav_links():
    """Regression test: link discovery must run on RAW html (pre-noise-
    strip), otherwise nav-only-linked pages become unreachable."""
    parsed = parse_html(SAMPLE_HTML, "https://coirei.com/pricing")
    assert "/pricing" in parsed.links
    assert "/" in parsed.links


def test_classify_homepage_variants():
    assert classify_page("https://example.com/") == "home"
    assert classify_page("https://example.com") == "home"
    assert classify_page("https://example.com/index.html") == "home"


def test_classify_pricing_page():
    assert classify_page("https://example.com/pricing", "Pricing Plans") == "pricing"


def test_classify_case_study_page():
    assert classify_page("https://example.com/case-studies/acme") == "case_study"


def test_dedupe_blocks_case_insensitive():
    blocks = ["Hello World", "hello world", "Different"]
    assert dedupe_blocks(blocks) == ["Hello World", "Different"]
