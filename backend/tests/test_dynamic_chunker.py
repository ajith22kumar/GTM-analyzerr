import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chunking.dynamic_chunker import (
    MAX_CHUNK_CHARS, MIN_SECTION_CHARS, chunk_page,
)
from app.chunking.section_classifier import classify_section
from app.extraction.html_parser import ContentBlock


def blocks(*pairs):
    return [ContentBlock(tag=t, text=x) for t, x in pairs]

def test_empty_input_returns_no_chunks():
    assert chunk_page("product", [], "https://x.com") == []

def test_single_section_becomes_one_chunk():
    page_blocks = blocks(
        ("h1", "Acme Flow"),
        ("p", "Acme Flow automates repetitive team workflows end to end."),
    )
    chunks = chunk_page("product", page_blocks, "https://acme.com/product")
    assert len(chunks) == 1
    assert "Acme Flow" in chunks[0].content
    assert chunks[0].metadata["url"] == "https://acme.com/product"
    assert chunks[0].metadata["page_type"] == "product"

def test_multiple_headings_produce_multiple_sections():
    page_blocks = blocks(
        ("h1", "Acme Flow"),
        ("p", "A" * 100),  # long enough to not get merged away
        ("h2", "Pricing"),
        ("p", "B" * 100),
        ("h2", "FAQ"),
        ("p", "C" * 100),
    )
    chunks = chunk_page("product", page_blocks, "https://acme.com/product")
    sections = [c.section for c in chunks]
    assert "pricing" in sections
    assert "faq" in sections
    assert len(chunks) == 3

def test_tiny_trailing_section_merges_into_previous():
    page_blocks = blocks(
        ("h2", "Pricing"),
        ("p", "Starter plan is nineteen dollars per month for one seat, billed annually with no setup fee."),
        ("h2", "FAQ"),
        ("p", "Yes."),  
    )
    chunks = chunk_page("product", page_blocks, "https://acme.com/product")
    assert len(chunks) == 1
    assert "Yes." in chunks[0].content
    assert "Starter plan" in chunks[0].content

def test_long_section_splits_into_multiple_chunks_with_overlap():
    long_paragraphs = [f"Paragraph number {i} contains some filler text to pad it out. " * 3 for i in range(20)]
    page_blocks = blocks(("h2", "Features"), *[("p", p) for p in long_paragraphs])
    chunks = chunk_page("product", page_blocks, "https://acme.com/product")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= MAX_CHUNK_CHARS + 200  
    assert all("Features" in c.content for c in chunks)

def test_single_oversized_paragraph_gets_hard_split():
    huge_paragraph = "word " * 1000  # ~5000 chars, way over MAX_CHUNK_CHARS
    page_blocks = blocks(("h2", "Overview"), ("p", huge_paragraph))
    chunks = chunk_page("product", page_blocks, "https://acme.com/product")
    assert len(chunks) > 1
    assert all(len(c.content) <= MAX_CHUNK_CHARS for c in chunks)

def test_content_before_first_heading_becomes_overview_section():
    page_blocks = blocks(
        ("p", "Intro paragraph with no heading above it at all here for padding."),
        ("h2", "Features"),
        ("p", "Some feature text goes here for padding purposes too."),
    )
    chunks = chunk_page("product", page_blocks, "https://acme.com/product")
    assert chunks[0].section == "overview"
    assert chunks[0].heading is None

def test_metadata_includes_section_and_extra_fields():
    page_blocks = blocks(("h2", "Pricing"), ("p", "Some pricing text that is long enough to stand alone here."))
    chunks = chunk_page(
        "product", page_blocks, "https://acme.com/product",
        extra_metadata={"company": "Acme"},
    )
    assert chunks[0].metadata["section"] == "pricing"
    assert chunks[0].metadata["company"] == "Acme"

def test_section_classifier_known_labels():
    assert classify_section("Pricing Plans") == "pricing"
    assert classify_section("Frequently Asked Questions") == "faq"
    assert classify_section("Our Mission") == "mission"
    assert classify_section("Meet the Team") == "leadership"
    assert classify_section(None) == "overview"
    assert classify_section("Something Totally Unrelated Xyz") == "other"
