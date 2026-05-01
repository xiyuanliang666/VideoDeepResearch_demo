"""
Evidence-tier policy (rsagent-v3): constants referenced by critic and research prompts.

Registry-tier facts require primary / official sources; travel and UGC sites must not be
upgraded to “verified registry” in synthesis.

Host / dataset may set an **evidence domain** (env ``RESEARCH_EVIDENCE_DOMAIN``) to bias
``deep_research_web_search`` worker ``site:`` filters and to align the Critic’s
``registry_verified`` expectations with that profile.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Evidence domain (B → A/D): normalize labels from .env / CLI / dataset
# ---------------------------------------------------------------------------

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
    """
    Return a canonical domain id for merging Serper worker sites and critic hints.

    Unknown non-empty strings are preserved lowercased (after strip); empty → ``general``.
    """
    s = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return "general"
    return _EVIDENCE_DOMAIN_ALIASES.get(s, s)


# Worker ``id`` (must match ``DEEP_RESEARCH_WORKER_CONFIGS``) → replacement ``sites`` list.
DOMAIN_WORKER_SITE_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "registry": {
        "FACT_CHECKER": ["whc.unesco.org", "ich.unesco.org", "unesco.org"],
    },
    "academic": {
        # Bias fact worker toward scholarly / standards context without removing other workers.
        "FACT_CHECKER": ["arxiv.org", "scholar.google.com", "ieee.org", "acm.org"],
    },
    "finance": {
        "FACT_CHECKER": ["sec.gov", "investor.gov", "federalregister.gov"],
    },
    "sports": {
        "FACT_CHECKER": ["olympics.com", "fifa.com", "worldathletics.org"],
    },
}


_CRITIC_DOMAIN_EXTRA: dict[str, str] = {
    "registry": (
        "The task profile is **registry-heavy** (e.g. UNESCO / official lists). When ranking URLs, "
        "**strongly prefer** whc.unesco.org / unesco.org (and similar primary government or "
        "standards sites). For **registry_verified: yes**, the supporting Jina body must be from "
        "such a **primary** domain—not travel blogs, TripAdvisor, or generic video descriptions."
    ),
    "academic": (
        "Bias toward **papers, preprints, and technical primary pages** when choosing Top-3 URLs. "
        "**registry_verified** is usually **n/a** unless the question is explicitly registry-tier; "
        "then still require a true primary registry source."
    ),
    "finance": (
        "Prefer **regulatory and issuer-primary** pages (e.g. SEC filings, official investor "
        "relations). Treat **registry_verified** as **n/a** unless the claim is a legal/registry "
        "listing; otherwise use **anchored_attribute** / **general_web** honestly."
    ),
    "sports": (
        "Prefer **official federation / games / league** sites when verifying facts. "
        "**registry_verified** is **n/a** unless the question asks for a legal or standards "
        "registry; do not treat fan wikis as official."
    ),
}


def critic_host_evidence_domain_block(normalized_domain: str) -> str:
    """Append to Critic rank/integrate user messages when domain is not ``general``."""
    dom = normalize_evidence_domain(normalized_domain)
    if dom == "general":
        return ""
    lines = _CRITIC_DOMAIN_EXTRA.get(dom)
    extra = lines if lines else (
        "Prefer **primary** sources that match this profile when scoring URLs and setting "
        "**registry_verified**; do not upgrade blogs or UGC to registry-tier **yes**."
    )
    return (
        f"\n\n**Host evidence_domain (from env / dataset):** `{dom}`\n"
        f"{extra}\n"
    )


# Domains often acceptable as **primary** evidence for registry-style claims (not exhaustive).
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

# Short block embedded in critic user prompts (English).
REGISTRY_SOURCE_GUIDANCE = (
    "**Registry-tier claims** (examples: a named UNESCO World Heritage *inscribed* property and its "
    "**official** full name; definitive ISO 3166-1/3166-2 codes as standardized identifiers; "
    "legal/government registry entries) may be treated as **verified** only when the supporting "
    "page body is from a **primary** source (e.g. whc.unesco.org list entries, iso.org, relevant "
    "national `.gov` / government gazette). "
    "**TripAdvisor, YouTube descriptions, Reddit, blogs, generic travel sites, and marketing copy** "
    "are **not** primary registry evidence — never state that a specific site is “a World Heritage "
    "Site” or give official ISO codes as **confirmed** from those alone."
)

ANCHORED_SEARCH_GUIDANCE = (
    "**Anchored-attribute** research: when the video already shows a **specific** company, person, "
    "product, or institution name, every `deep_research_web_search` query for attributes (founding year, "
    "official robot name, etc.) should **include that anchor string** so results are not confused with "
    "homonyms."
)

HIGH_AMBIGUITY_SEARCH_GUIDANCE = (
    "**High-ambiguity** topics (same event name in multiple cities/years; colloquial place names; "
    "several UNESCO-related landscapes in one region): do **not** rely on a **single** broad search. "
    "Plan **multiple** `deep_research_web_search` calls with **different hypotheses** (alternate "
    "cities, official inscription title vs. tourist attraction name) or query the **registry** "
    "first (e.g. site + “world heritage” + `whc.unesco.org`) before tying a label to one landmark."
)

DISAMBIGUATION_GUIDANCE = (
    "**On-screen vs. official naming**: if visible text looks like a **project code, campaign, or "
    "subtitle** (e.g. on a robot chest) while the question asks for the **official name**, the final "
    "answer must **briefly separate** layers (what appears on screen vs. what authoritative sources "
    "call the entity)."
)
