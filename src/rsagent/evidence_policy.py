"""Evidence-tier policy: domain normalization and worker site overrides (from rsagent-v3)."""

from __future__ import annotations

_EVIDENCE_DOMAIN_ALIASES: dict[str, str] = {
    "": "general", "default": "general", "general": "general", "any": "general",
    "registry": "registry", "heritage": "registry", "unesco": "registry",
    "world_heritage": "registry", "world-heritage": "registry", "whs": "registry",
    "academic": "academic", "science": "academic", "tech": "academic", "paper": "academic",
    "finance": "finance", "sec": "finance", "investing": "finance",
    "sports": "sports", "olympics": "sports",
}


def normalize_evidence_domain(raw: str | None) -> str:
    s = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return "general"
    return _EVIDENCE_DOMAIN_ALIASES.get(s, s)


DOMAIN_WORKER_SITE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "registry": {"FACT_CHECKER": ["whc.unesco.org", "ich.unesco.org", "unesco.org"]},
    "academic": {"FACT_CHECKER": ["arxiv.org", "scholar.google.com", "ieee.org", "acm.org"]},
    "finance": {"FACT_CHECKER": ["sec.gov", "investor.gov", "federalregister.gov"]},
    "sports": {"FACT_CHECKER": ["olympics.com", "fifa.com", "worldathletics.org"]},
}
