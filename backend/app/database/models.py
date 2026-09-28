from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _uuid_col():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Website(Base):
    __tablename__ = "websites"

    id = _uuid_col()
    domain = Column(String(255), nullable=False, index=True)
    url = Column(String(2048), nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    pages_discovered = Column(Integer, nullable=False, default=0)
    pages_crawled = Column(Integer, nullable=False, default=0)
    pages_failed = Column(Integer, nullable=False, default=0)
    sitemap_found = Column(Integer, nullable=False, default=0)  # 0/1 boolean
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    pages = relationship("Page", back_populates="website", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="website", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="website", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("url", name="uq_websites_url"),)


class Page(Base):
    __tablename__ = "pages"

    id = _uuid_col()
    website_id = Column(UUID(as_uuid=True), ForeignKey("websites.id", ondelete="CASCADE"), nullable=False, index=True)
    url = Column(String(2048), nullable=False)
    canonical_url = Column(String(2048), nullable=True)
    title = Column(String(1024), nullable=True)
    meta_description = Column(Text, nullable=True)
    h1 = Column(String(1024), nullable=True)
    headings = Column(JSON, nullable=True)          # list[str]
    main_content = Column(Text, nullable=True)
    content_blocks = Column(JSON, nullable=True)     # list[{"tag": str, "text": str}], for chunking
    page_type = Column(String(64), nullable=False, default="other", index=True)
    language = Column(String(16), nullable=True)
    status_code = Column(Integer, nullable=True)
    depth = Column(Integer, nullable=False, default=0)
    crawl_status = Column(String(32), nullable=False, default="ok")  # ok | fetch_failed | http_4xx | http_5xx | non_html
    error = Column(String(255), nullable=True)
    crawl_timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    website = relationship("Website", back_populates="pages")
    chunks = relationship("Chunk", back_populates="page", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("website_id", "url", name="uq_pages_website_url"),)


class Chunk(Base):
    __tablename__ = "chunks"

    id = _uuid_col()
    page_id = Column(UUID(as_uuid=True), ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    section = Column(String(128), nullable=True)   
    embedding = Column(Vector(1536), nullable=True)
    chunk_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    page = relationship("Page", back_populates="chunks")


class Analysis(Base):
    __tablename__ = "analyses"

    id = _uuid_col()
    website_id = Column(UUID(as_uuid=True), ForeignKey("websites.id", ondelete="CASCADE"), nullable=False, index=True)
    analysis_type = Column(String(64), nullable=False)  # company | product | gtm | market | competitor | channels
    result = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    website = relationship("Website", back_populates="analyses")

class Report(Base):
    __tablename__ = "reports"

    id = _uuid_col()
    website_id = Column(UUID(as_uuid=True), ForeignKey("websites.id", ondelete="CASCADE"), nullable=False, index=True)
    report_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    website = relationship("Website", back_populates="reports")
