from __future__ import annotations

SECTION_LABELS = (
    "overview", "features", "benefits", "use_cases", "pricing",
    "technical_details", "faq", "mission", "vision", "leadership",
    "locations", "history", "problem", "solution", "implementation",
    "results", "metrics", "customer", "other",
)

_SECTION_SIGNALS: list[tuple[str, list[str]]] = [
    ("pricing", ["pricing", "plan", "price", "cost", "tier", "subscription"]),
    ("faq", ["faq", "frequently asked", "questions"]),
    ("technical_details", ["technical", "specification", "spec", "architecture", "api", "integration"]),
    ("use_cases", ["use case", "use-case", "how it works", "workflow"]),
    ("benefits", ["benefit", "why choose", "why us", "value"]),
    ("features", ["feature", "capabilit", "what you get", "included"]),
    ("mission", ["mission"]),
    ("vision", ["vision"]),
    ("leadership", ["leadership", "team", "founder", "executive", "management"]),
    ("locations", ["location", "office", "headquarters", "address", "where we"]),
    ("history", ["history", "our story", "timeline", "journey", "founded"]),
    ("problem", ["problem", "challenge", "pain point"]),
    ("solution", ["solution", "approach"]),
    ("implementation", ["implementation", "rollout", "deployment", "how we did it"]),
    ("results", ["result", "outcome", "impact", "roi"]),
    ("metrics", ["metric", "kpi", "increase", "decrease", "%", "percent"]),
    ("customer", ["customer", "client", "case stud", "testimonial", "success stor"]),
]


def classify_section(heading_text: str | None) -> str:
    if not heading_text:
        return "overview"
    lowered = heading_text.lower()
    for label, keywords in _SECTION_SIGNALS:
        if any(kw in lowered for kw in keywords):
            return label
    return "other"
