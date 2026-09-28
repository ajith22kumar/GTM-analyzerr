from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl, field_validator

class CrawlRequest(BaseModel):
    url: HttpUrl
    max_pages: Optional[int] = None
    max_depth: Optional[int] = None

    @field_validator("max_pages")
    @classmethod
    def validate_max_pages(cls, v):
        if v is not None and (v < 1 or v > 5000):
            raise ValueError("max_pages must be between 1 and 5000")
        return v

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, v):
        if v is not None and (v < 0 or v > 50):
            raise ValueError("max_depth must be between 0 and 50")
        return v

class WebsiteStatusResponse(BaseModel):
    id: UUID
    domain: str
    url: str
    status: str
    pages_discovered: int
    pages_crawled: int
    pages_failed: int
    sitemap_found: bool
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}

class PageSummary(BaseModel):
    id: UUID
    url: str
    title: Optional[str] = None
    page_type: str
    status_code: Optional[int] = None
    crawl_status: str
    model_config = {"from_attributes": True}

class PageListResponse(BaseModel):
    website_id: UUID
    total: int
    pages: list[PageSummary]

class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and (v < 1 or v > 20):
            raise ValueError("top_k must be between 1 and 20")
        return v

class RetrievedChunkResponse(BaseModel):
    content: str
    section: str
    url: str
    page_type: str
    similarity: float

class ChatResponse(BaseModel):
    question: str
    results: list[RetrievedChunkResponse]
