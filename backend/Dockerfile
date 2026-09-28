FROM python:3.12-slim

WORKDIR /app

# System deps: lxml needs libxml2/libxslt at build time; Playwright's
# browser binaries need a handful of shared libs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only install Chromium (not all 3 browsers) to keep the image smaller.
# Playwright is only invoked for JS-heavy pages (Section 2) — most sites
# never need it, but it must be present when the crawler falls back to it.
RUN playwright install --with-deps chromium

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
