"""Evidence-domain policy shared by web tools and critics.

This module keeps source-policy and evidence-tier rules. It biases search
workers and critic prompts without requiring benchmark code to know the
underlying search implementation.
"""

from __future__ import annotations

_EVIDENCE_DOMAIN_ALIASES: dict[str, str] = {
    "": "general",
    "default": "general",
    "general": "general",
    "any": "general",
    "registry": "registry",
    "heritage": "registry",
    "unesco": "registry",
    "world_heritage": "registry",
    "world-heritage": "registry",
    "whs": "registry",
    "academic": "academic",
    "science": "academic",
    "tech": "academic",
    "paper": "academic",
    "finance": "finance",
    "sec": "finance",
    "investing": "finance",
    "sports": "sports",
    "olympics": "sports",
}


def normalize_evidence_domain(raw: str | None) -> str:
    """Normalize user/config labels to a canonical evidence-domain id."""
    text = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not text:
        return "general"
    return _EVIDENCE_DOMAIN_ALIASES.get(text, text)


DOMAIN_WORKER_SITE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "registry": {
        "FACT_CHECKER": ["whc.unesco.org", "ich.unesco.org", "unesco.org"],
    },
    "academic": {
        "FACT_CHECKER": ["arxiv.org", "scholar.google.com", "ieee.org", "acm.org"],
    },
    "finance": {
        "FACT_CHECKER": ["sec.gov", "investor.gov", "federalregister.gov"],
    },
    "sports": {
        "FACT_CHECKER": ["olympics.com", "fifa.com", "worldathletics.org"],
    },
}


REGISTRY_PRIMARY_DOMAIN_SUBSTRINGS: tuple[str, ...] = (
    "whc.unesco.org",
    "unesco.org",
    "iso.org",
    "iho.int",
    "un.org",
    "europa.eu",
    "gov.",
    "go.jp",
    "go.kr",
    "gouv.",
)


def is_registry_primary_url(url: str) -> bool:
    """Return whether a URL can plausibly support registry-tier verification."""
    lowered = (url or "").lower()
    return any(domain in lowered for domain in REGISTRY_PRIMARY_DOMAIN_SUBSTRINGS)


def critic_host_evidence_domain_block(normalized_domain: str) -> str:
    """Short hint that can be appended to critic/ranker prompts."""
    domain = normalize_evidence_domain(normalized_domain)
    if domain == "general":
        return ""
    if domain == "registry":
        extra = (
            "Prefer official registry or government sources. Do not upgrade blogs, "
            "travel sites, UGC, or generic descriptions to registry-verified evidence."
        )
    elif domain == "sports":
        extra = "Prefer official federation, league, tournament, or club sources."
    elif domain == "academic":
        extra = "Prefer papers, preprints, official project pages, standards, and technical primary sources."
    elif domain == "finance":
        extra = "Prefer regulatory filings, government pages, and official issuer pages."
    else:
        extra = "Prefer primary or authoritative sources matching this evidence profile."
    return f"\n\nHost evidence_domain: `{domain}`\n{extra}\n"
