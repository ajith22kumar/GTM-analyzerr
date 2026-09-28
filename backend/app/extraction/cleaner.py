from __future__ import annotations

import re

from bs4 import BeautifulSoup

NOISE_TAGS = [
    "script", "style", "noscript", "nav", "footer",
    "iframe", "svg", "form", "button", "aside",
]

NOISE_SELECTORS = [
    "cookie", "consent", "gdpr", "banner", "newsletter-popup",
    "modal", "popup", "sidebar", "breadcrumb", "pagination",
    "social-share", "share-buttons", "advert", "sponsor", "site-nav",
    "mobile-menu", "skip-link", "back-to-top",
]

WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def strip_noise_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for selector_fragment in NOISE_SELECTORS:
        for tag in soup.find_all(
            attrs={"class": re.compile(selector_fragment, re.I)}
        ):
            tag.decompose()
        for tag in soup.find_all(
            attrs={"id": re.compile(selector_fragment, re.I)}
        ):
            tag.decompose()

    for comment in soup.find_all(string=lambda s: s.__class__.__name__ == "Comment"):
        comment.extract()

    return str(soup)


def normalize_whitespace(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def dedupe_blocks(blocks: list[str], min_len: int = 2) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for block in blocks:
        cleaned = normalize_whitespace(block)
        if len(cleaned) < min_len:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def looks_like_boilerplate_line(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return True
    patterns = (
        "all rights reserved",
        "privacy policy",
        "terms of service",
        "terms & conditions",
        "cookie policy",
        "©",
        "accept all cookies",
        "we use cookies",
    )
    if any(p in lowered for p in patterns) and len(lowered) < 120:
        return True
    return False


def clean_text_blocks(blocks: list[str]) -> list[str]:
    deduped = dedupe_blocks(blocks)
    return [b for b in deduped if not looks_like_boilerplate_line(b)]


def clean_structured_blocks(blocks: list) -> list:
    seen: set[str] = set()
    result = []
    for block in blocks:
        cleaned = normalize_whitespace(block.text)
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if looks_like_boilerplate_line(cleaned):
            continue
        seen.add(key)
        block.text = cleaned
        result.append(block)
    return result
