from __future__ import annotations
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

@dataclass
class ContentBlock:
    tag: str   # "h1" | "h2" | "h3" | "h4" | "h5" | "p" | "li" | "td" | "blockquote" | "figcaption"
    text: str

@dataclass
class ParsedPage:
    url: str
    title: str = ""
    meta_description: str = ""
    language: str = ""
    h1: str = ""
    headings: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    raw_text_blocks: list[str] = field(default_factory=list)
    structured_blocks: list[ContentBlock] = field(default_factory=list)

def parse_html(html: str, url: str) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag["content"].strip()

    language = ""
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        language = html_tag["lang"].strip()

    h1_text = ""
    h1_tag = soup.find("h1")
    if h1_tag:
        h1_text = h1_tag.get_text(strip=True)

    headings: list[str] = []
    for level in ("h1", "h2", "h3", "h4"):
        for tag in soup.find_all(level):
            text = tag.get_text(strip=True)
            if text:
                headings.append(text)

    links: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if href:
            links.append(href)

    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("body")
        or soup
    )

    text_blocks: list[str] = []
    structured_blocks: list[ContentBlock] = []
    for tag in content_root.find_all(
        ["p", "li", "h1", "h2", "h3", "h4", "h5", "td", "blockquote", "figcaption"]
    ):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            text_blocks.append(text)
            structured_blocks.append(ContentBlock(tag=tag.name, text=text))

    return ParsedPage(
        url=url,
        title=title,
        meta_description=meta_description,
        language=language,
        h1=h1_text,
        headings=headings,
        links=links,
        raw_text_blocks=text_blocks,
        structured_blocks=structured_blocks,
    )
