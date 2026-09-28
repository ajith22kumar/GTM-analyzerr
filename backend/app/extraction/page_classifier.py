from __future__ import annotations
from urllib.parse import urlparse
PAGE_TYPES = (
    "home", "about", "product", "service", "pricing", "case_study",
    "customer", "blog", "contact", "careers", "partner", "faq", "other",
)

_SIGNALS: list[tuple[str, list[str]]] = [
    ("pricing", ["pricing", "plans", "price", "cost"]),
    ("case_study", ["case-study", "case study", "case-studies", "success-stor", "customer-stor"]),
    ("customer", ["customers", "clients", "testimonial", "who-we-serve"]),
    ("product", ["product", "platform", "solutions", "features"]),
    ("service", ["services", "what-we-do"]),
    ("about", ["about", "about-us", "company", "our-story", "mission", "team", "leadership"]),
    ("careers", ["careers", "jobs", "join-us", "hiring"]),
    ("partner", ["partners", "partnership", "alliances", "integrations"]),
    ("faq", ["faq", "frequently-asked", "help", "support"]),
    ("blog", ["blog", "news", "articles", "insights", "resources"]),
    ("contact", ["contact", "get-in-touch", "reach-us"]),
]

def classify_page(url: str, title: str = "", headings: list[str] | None = None) -> str:
    headings = headings or []
    path = urlparse(url).path.lower()

    if path in ("", "/", "/index.html", "/index.php", "/index.htm", "/home"):
        return "home"

    haystack = " ".join([path, title.lower(), " ".join(h.lower() for h in headings)])

    scores: dict[str, int] = {}
    for page_type, keywords in _SIGNALS:
        score = sum(1 for kw in keywords if kw in haystack)
        path_bonus = sum(2 for kw in keywords if kw in path)
        if score or path_bonus:
            scores[page_type] = score + path_bonus

    if not scores:
        return "other"

    return max(scores, key=scores.get)
