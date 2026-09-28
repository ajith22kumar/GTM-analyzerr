from __future__ import annotations

from dataclasses import dataclass, field

from app.chunking.section_classifier import classify_section
from app.extraction.html_parser import ContentBlock

MAX_CHUNK_CHARS = 1500
MIN_SECTION_CHARS = 60
CHUNK_OVERLAP_CHARS = 150
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5"}

@dataclass
class TextChunk:
    chunk_index: int
    content: str
    section: str
    heading: str | None
    metadata: dict = field(default_factory=dict)

@dataclass
class _RawSection:
    heading: str | None
    label: str
    paragraphs: list[str]

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.paragraphs) + len(self.heading or "")

    @property
    def text(self) -> str:
        parts = [self.heading] if self.heading else []
        parts.extend(self.paragraphs)
        return "\n".join(p for p in parts if p)

def _group_into_sections(blocks: list[ContentBlock]) -> list[_RawSection]:
    sections: list[_RawSection] = []
    current = _RawSection(heading=None, label="overview", paragraphs=[])

    for block in blocks:
        if block.tag in HEADING_TAGS:
            if current.paragraphs or current.heading:
                sections.append(current)
            current = _RawSection(
                heading=block.text, label=classify_section(block.text), paragraphs=[]
            )
        else:
            current.paragraphs.append(block.text)

    if current.paragraphs or current.heading:
        sections.append(current)
    return sections

def _merge_tiny_sections(sections: list[_RawSection]) -> list[_RawSection]:
    
    if not sections:
        return sections

    merged: list[_RawSection] = []
    pending: _RawSection | None = None

    for section in sections:
        if pending is not None:
            combined_chars = pending.char_count + section.char_count
            if pending.char_count < MIN_SECTION_CHARS and combined_chars <= MAX_CHUNK_CHARS:
                section = _RawSection(
                    heading=pending.heading or section.heading,
                    label=pending.label if pending.label != "overview" else section.label,
                    paragraphs=pending.paragraphs + ([section.heading] if section.heading else []) + section.paragraphs,
                )
            else:
                merged.append(pending)
            pending = None
        pending = section

    if pending is not None:
        if merged and pending.char_count < MIN_SECTION_CHARS:
            last = merged[-1]
            if last.char_count + pending.char_count <= MAX_CHUNK_CHARS:
                merged[-1] = _RawSection(
                    heading=last.heading,
                    label=last.label,
                    paragraphs=last.paragraphs + ([pending.heading] if pending.heading else []) + pending.paragraphs,
                )
            else:
                merged.append(pending)
        else:
            merged.append(pending)

    return merged

def _split_long_section(section: _RawSection) -> list[str]:
    if section.char_count <= MAX_CHUNK_CHARS:
        return [section.text]

    pieces: list[str] = []
    current_parts: list[str] = [section.heading] if section.heading else []
    current_len = len(section.heading or "")

    for para in section.paragraphs:
        if current_len + len(para) > MAX_CHUNK_CHARS and current_parts:
            pieces.append("\n".join(current_parts))
            overlap_source = current_parts[-1]
            overlap = overlap_source[-CHUNK_OVERLAP_CHARS:] if len(overlap_source) > CHUNK_OVERLAP_CHARS else overlap_source
            current_parts = ([section.heading] if section.heading else []) + [f"...{overlap}"]
            current_len = len(section.heading or "") + len(overlap)

        if len(para) > MAX_CHUNK_CHARS:
            for start in range(0, len(para), MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS):
                piece = para[start:start + MAX_CHUNK_CHARS]
                pieces.append(piece)
            current_parts = [section.heading] if section.heading else []
            current_len = len(section.heading or "")
            continue

        current_parts.append(para)
        current_len += len(para)

    if current_parts and any(p for p in current_parts if p != section.heading):
        pieces.append("\n".join(current_parts))
    elif not pieces:
        pieces.append("\n".join(current_parts))

    return pieces


def chunk_page(
    page_type: str,
    structured_blocks: list[ContentBlock],
    url: str,
    extra_metadata: dict | None = None,
) -> list[TextChunk]:
    if not structured_blocks:
        return []

    sections = _group_into_sections(structured_blocks)
    sections = _merge_tiny_sections(sections)

    chunks: list[TextChunk] = []
    base_metadata = {"url": url, "page_type": page_type, **(extra_metadata or {})}

    for section in sections:
        if not section.text.strip():
            continue
        pieces = _split_long_section(section)
        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    content=piece,
                    section=section.label,
                    heading=section.heading,
                    metadata={**base_metadata, "section": section.label},
                )
            )

    return chunks
